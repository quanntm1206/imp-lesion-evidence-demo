"""Receipt-gated Loop206 inference with exact fixed-cache fallback."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from lesion_robustness.demo.fixed_cache import (
    ProductionFixedSampleProvider,
    _AuthorizedFixedSample,
    _build_production_provider,
)
from lesion_robustness.demo.geometry import overlay_mask, prepare_image, restore_probability
from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.demo.loop206_prior import load_deployment_prior_with_receipt_hash
from lesion_robustness.preprocessing import preprocess_image_from_config


REGISTRY_SCHEMA = "loop206.demo.models.v1"
DEFAULT_SEED = 206
_INFERENCE_LOCK = Lock()
_AUTHORIZATION_TOKEN = object()
_REGISTRY_TOKEN = object()
PINNED_REGISTRY = {
    "schema_version": REGISTRY_SCHEMA,
    "control": {
        "model_id": "L206-control-s206",
        "checkpoint_env": "IMP_LOOP206_CONTROL_CHECKPOINT",
        "checkpoint_sha256": "be606b0a0940839b019ea60117dda4b27f9b8f04d54306b5b676f2c29516fcef",
    },
    "candidate": {
        "model_id": "L206-contour-channel-s206",
        "checkpoint_env": "IMP_LOOP206_CANDIDATE_CHECKPOINT",
        "checkpoint_sha256": "afb86b2a5161189369dbc3c985e78f214c305470661048c6643726612f57638b",
    },
    "prior_env": "IMP_LOOP206_PRIOR",
    "prior_receipt_env": "IMP_LOOP206_PRIOR_RECEIPT",
    "prior_receipt_sha256": None,
}


class CandidateUnavailableError(RuntimeError):
    """Raised before inference when arbitrary candidate use lacks authorization."""


@runtime_checkable
class ModelEndpoint(Protocol):
    model_id: str
    checkpoint_sha256: str
    device: str

    def predict_logits(self, batch: np.ndarray) -> np.ndarray: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class ReceiptAuthorizedPrior:
    prior: Any
    receipt_sha256: str
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _AUTHORIZATION_TOKEN:
            raise TypeError("ReceiptAuthorizedPrior must come from the receipt-bound loader")


@dataclass(frozen=True)
class ComparisonResult:
    original_rgb: np.ndarray
    control_probability: np.ndarray
    candidate_probability: np.ndarray
    control_mask: np.ndarray
    candidate_mask: np.ndarray
    control_overlay: np.ndarray
    candidate_overlay: np.ndarray
    control_latency_ms: float
    candidate_latency_ms: float
    total_latency_ms: float
    device: str
    control_model_id: str
    candidate_model_id: str
    control_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    prior_receipt_sha256: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ControlOnlyResult:
    mode: str
    original_rgb: np.ndarray
    control_probability: np.ndarray
    control_mask: np.ndarray
    control_overlay: np.ndarray
    control_latency_ms: float
    device: str
    control_model_id: str
    control_checkpoint_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedModelRegistry:
    control: ModelEndpoint
    candidate: ModelEndpoint
    preprocessing: dict[str, Any]
    corruption_configs: dict[str, Any]
    seed: int
    prior: ReceiptAuthorizedPrior | None
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _REGISTRY_TOKEN:
            raise TypeError("LoadedModelRegistry must come from the pinned registry loader")

    def build_fixed_provider(
        self,
        *,
        dataset_index: str | Path,
        dataset_roots: list[str | Path] | tuple[str | Path, ...],
        candidate_manifest: str | Path,
        zero_manifest: str | Path,
        live_config: str | Path,
    ) -> ProductionFixedSampleProvider:
        return _build_production_provider(
            dataset_index=dataset_index,
            dataset_roots=dataset_roots,
            candidate_manifest=candidate_manifest,
            zero_manifest=zero_manifest,
            live_config=live_config,
            registry_preprocessing=self.preprocessing,
            corruption_configs=self.corruption_configs,
            project_seed=self.seed,
        )

    def build_service(
        self, *, fixed_provider: ProductionFixedSampleProvider | None = None
    ) -> "Loop206ComparisonService":
        config = deepcopy(self.preprocessing)
        return Loop206ComparisonService(
            self.control,
            self.candidate,
            self.prior,
            fixed_provider=fixed_provider,
            preprocessor=lambda image: preprocess_image_from_config(
                image, deepcopy(config)
            ),
        )


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt_authorized_prior(
    artifact_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str,
) -> ReceiptAuthorizedPrior:
    prior, receipt_hash = load_deployment_prior_with_receipt_hash(
        artifact_path,
        receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return ReceiptAuthorizedPrior(prior, receipt_hash, _AUTHORIZATION_TOKEN)


def _validate_model_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (384, 384, 3) or value.dtype != np.uint8:
        raise ValueError("Loop206 model RGB must be uint8 with shape (384, 384, 3)")
    return np.ascontiguousarray(value)


def _validate_channel(channel: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(channel)
    if value.shape != (384, 384):
        raise ValueError(f"Loop206 {label} channel geometry mismatch")
    if value.dtype == np.bool_:
        value = value.astype(np.uint8) * 255
    if value.dtype != np.uint8:
        raise ValueError(f"Loop206 {label} channel must use uint8 pixels")
    return np.ascontiguousarray(value)


def _model_batch(rgb: np.ndarray, channel: np.ndarray) -> np.ndarray:
    image = _validate_model_rgb(rgb).astype(np.float32) / 255.0
    extra = _validate_channel(channel, label="extra").astype(np.float32) / 255.0
    return np.concatenate([image, extra[..., None]], axis=2).transpose(2, 0, 1)[None]


def _sigmoid_logits(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float32)
    if value.shape != (1, 1, 384, 384) or not np.isfinite(value).all():
        raise ValueError("Loop206 model must return finite logits with shape (1, 1, 384, 384)")
    clipped = np.clip(value[0, 0], -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def _fixed_public_metadata(fixed: Any) -> dict[str, Any]:
    return {
        "comparison_source": "exact_fixed_cache",
        "group_key": str(fixed.group_key),
        "sample_id": str(fixed.sample_id),
        "corruption": str(fixed.corruption),
        "candidate_manifest_sha256": str(fixed.candidate_manifest_sha256),
        "candidate_data_sha256": str(fixed.candidate_data_sha256),
        "zero_manifest_sha256": str(fixed.zero_manifest_sha256),
        "zero_data_sha256": str(fixed.zero_data_sha256),
        "mask_sha256_raw": str(fixed.mask_sha256_raw),
        "mask_sha256_binary": str(fixed.mask_sha256_binary),
        "mask_sha256_runtime": str(fixed.mask_sha256_runtime),
        "historical_cache_provenance_drift": bool(
            fixed.historical_cache_provenance_drift
        ),
    }


def _inference_context(control: ModelEndpoint, candidate: ModelEndpoint):
    if isinstance(control, TorchModelEndpoint) and isinstance(candidate, TorchModelEndpoint):
        import torch

        return torch.inference_mode()
    return nullcontext()


class Loop206ComparisonService:
    def __init__(
        self,
        control: ModelEndpoint,
        candidate: ModelEndpoint,
        prior: ReceiptAuthorizedPrior | None = None,
        *,
        fixed_provider: ProductionFixedSampleProvider | None = None,
        preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not isinstance(control, ModelEndpoint) or not isinstance(candidate, ModelEndpoint):
            raise TypeError("Loop206 endpoints do not satisfy the inference contract")
        if control.device != candidate.device:
            raise ValueError("Loop206 model pair must use one device")
        if prior is not None and type(prior) is not ReceiptAuthorizedPrior:
            raise TypeError("Loop206 prior must be an exact ReceiptAuthorizedPrior")
        if fixed_provider is not None and (
            type(fixed_provider) is not ProductionFixedSampleProvider
            or not fixed_provider.is_production_authorized
        ):
            raise TypeError("Loop206 requires a production-authorized fixed provider")
        self.control = control
        self.candidate = candidate
        self.prior = prior
        self.fixed_provider = fixed_provider
        self.preprocessor = preprocessor or (lambda image: image)
        self.threshold = float(threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Loop206 threshold must be within (0, 1)")

    def _run_pair(
        self, control_batch: np.ndarray, candidate_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        started = perf_counter()
        with _INFERENCE_LOCK, _inference_context(self.control, self.candidate):
            control_started = perf_counter()
            control_logits = self.control.predict_logits(control_batch)
            self.control.synchronize()
            control_ms = (perf_counter() - control_started) * 1000.0
            candidate_started = perf_counter()
            candidate_logits = self.candidate.predict_logits(candidate_batch)
            self.candidate.synchronize()
            candidate_ms = (perf_counter() - candidate_started) * 1000.0
        total_ms = (perf_counter() - started) * 1000.0
        return (
            _sigmoid_logits(control_logits),
            _sigmoid_logits(candidate_logits),
            control_ms,
            candidate_ms,
            total_ms,
        )

    def _comparison_result(
        self,
        *,
        original_rgb: np.ndarray,
        original_shape: tuple[int, int],
        control_probability: np.ndarray,
        candidate_probability: np.ndarray,
        control_ms: float,
        candidate_ms: float,
        total_ms: float,
        prior_receipt_sha256: str | None,
        metadata: dict[str, Any],
    ) -> ComparisonResult:
        restored_control = restore_probability(control_probability, original_shape)
        restored_candidate = restore_probability(candidate_probability, original_shape)
        control_mask = (restored_control >= self.threshold).astype(np.uint8)
        candidate_mask = (restored_candidate >= self.threshold).astype(np.uint8)
        return ComparisonResult(
            original_rgb=original_rgb,
            control_probability=restored_control,
            candidate_probability=restored_candidate,
            control_mask=control_mask,
            candidate_mask=candidate_mask,
            control_overlay=overlay_mask(original_rgb, control_mask),
            candidate_overlay=overlay_mask(original_rgb, candidate_mask),
            control_latency_ms=float(control_ms),
            candidate_latency_ms=float(candidate_ms),
            total_latency_ms=float(total_ms),
            device=self.control.device,
            control_model_id=self.control.model_id,
            candidate_model_id=self.candidate.model_id,
            control_checkpoint_sha256=self.control.checkpoint_sha256,
            candidate_checkpoint_sha256=self.candidate.checkpoint_sha256,
            prior_receipt_sha256=prior_receipt_sha256,
            metadata=dict(metadata),
        )

    def compare(self, image: np.ndarray) -> ComparisonResult:
        if self.prior is None:
            raise CandidateUnavailableError(
                "arbitrary candidate comparison requires a receipt-authorized deployment prior"
            )
        prepared = prepare_image(image)
        model_rgb = _validate_model_rgb(self.preprocessor(prepared.model_rgb.copy()))
        candidate_channel = _validate_channel(
            self.prior.prior.predict(prepared.model_rgb.copy()), label="candidate"
        )
        zero_channel = np.zeros((384, 384), dtype=np.uint8)
        predictions = self._run_pair(
            _model_batch(model_rgb, zero_channel),
            _model_batch(model_rgb, candidate_channel),
        )
        return self._comparison_result(
            original_rgb=prepared.original_rgb,
            original_shape=prepared.original_shape,
            control_probability=predictions[0],
            candidate_probability=predictions[1],
            control_ms=predictions[2],
            candidate_ms=predictions[3],
            total_ms=predictions[4],
            prior_receipt_sha256=self.prior.receipt_sha256,
            metadata={"comparison_source": "receipt_authorized_prior"},
        )

    def compare_fixed(
        self,
        identifier: str,
        *,
        corruption: str,
    ) -> ComparisonResult:
        if self.fixed_provider is None:
            raise CandidateUnavailableError(
                "fixed comparison requires a production fixed provider"
            )
        fixed = self.fixed_provider.authorize(identifier, corruption=corruption)
        if type(fixed) is not _AuthorizedFixedSample:
            raise CandidateUnavailableError("fixed sample authorization is invalid")
        prepared = prepare_image(fixed.original_rgb)
        model_rgb = _validate_model_rgb(fixed.model_rgb)
        control_channel = _validate_channel(fixed.control_channel, label="control")
        candidate_channel = _validate_channel(fixed.candidate_channel, label="candidate")
        if np.any(control_channel):
            raise ValueError("Loop206 fixed comparison control channel must be all-zero")
        predictions = self._run_pair(
            _model_batch(model_rgb, control_channel),
            _model_batch(model_rgb, candidate_channel),
        )
        return self._comparison_result(
            original_rgb=prepared.original_rgb,
            original_shape=prepared.original_shape,
            control_probability=predictions[0],
            candidate_probability=predictions[1],
            control_ms=predictions[2],
            candidate_ms=predictions[3],
            total_ms=predictions[4],
            prior_receipt_sha256=None,
            metadata=_fixed_public_metadata(fixed),
        )

    def preview_control(self, image: np.ndarray) -> ControlOnlyResult:
        prepared = prepare_image(image)
        model_rgb = _validate_model_rgb(self.preprocessor(prepared.model_rgb.copy()))
        batch = _model_batch(model_rgb, np.zeros((384, 384), dtype=np.uint8))
        started = perf_counter()
        context = (
            _inference_context(self.control, self.candidate)
            if isinstance(self.control, TorchModelEndpoint)
            else nullcontext()
        )
        with _INFERENCE_LOCK, context:
            logits = self.control.predict_logits(batch)
            self.control.synchronize()
        latency = (perf_counter() - started) * 1000.0
        probability = restore_probability(_sigmoid_logits(logits), prepared.original_shape)
        mask = (probability >= self.threshold).astype(np.uint8)
        return ControlOnlyResult(
            mode="control_only",
            original_rgb=prepared.original_rgb,
            control_probability=probability,
            control_mask=mask,
            control_overlay=overlay_mask(prepared.original_rgb, mask),
            control_latency_ms=float(latency),
            device=self.control.device,
            control_model_id=self.control.model_id,
            control_checkpoint_sha256=self.control.checkpoint_sha256,
            metadata={"result_type": "arbitrary_image_control_only"},
        )


class TorchModelEndpoint:
    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        checkpoint_sha256: str,
        device: str,
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = device

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        result = output.detach().float().cpu().numpy()
        del output, tensor
        return result

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)


def _build_loop206_model() -> Any:
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn

    class _Loop206Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False)
            self.model = smp.Unet(
                encoder_name="mit_b3",
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            with torch.no_grad():
                self.input_adapter.weight.zero_()
                self.input_adapter.weight[0, 0, 0, 0] = 1.0
                self.input_adapter.weight[1, 1, 0, 0] = 1.0
                self.input_adapter.weight[2, 2, 0, 0] = 1.0

        def forward(self, value):
            return self.model(self.input_adapter(value))

    return _Loop206Model()


def _validate_checkpoint_config(
    config: Mapping[str, Any], *, model_id: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    project = config.get("project")
    model = config.get("model")
    benchmark = config.get("benchmark")
    expected_model = {
        "name": "segformer_mit",
        "smp_encoder": "mit_b3",
        "in_channels": 3,
        "input_channels": 4,
        "encoder_in_channels": 3,
        "out_channels": 1,
        "edge_aux": False,
    }
    if not isinstance(project, Mapping) or int(project.get("seed", -1)) != int(seed):
        raise ValueError("Loop206 checkpoint seed mismatch")
    if not isinstance(benchmark, Mapping) or benchmark.get("member_id") != model_id:
        raise ValueError("Loop206 checkpoint model ID mismatch")
    if not isinstance(model, Mapping) or any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("Loop206 checkpoint model architecture mismatch")
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Loop206 checkpoint preprocessing contract mismatch")
    robustness = config.get("robustness")
    corruptions = robustness.get("corruptions") if isinstance(robustness, Mapping) else None
    if not isinstance(corruptions, dict):
        raise ValueError("Loop206 checkpoint corruption contract mismatch")
    return deepcopy(preprocessing), deepcopy(corruptions)


def _validate_preprocessing_pair(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    control_copy = deepcopy(dict(control))
    candidate_copy = deepcopy(dict(candidate))
    expected_suffixes = {
        "control": "pilot_cache_v2_zero_control/manifest.json",
        "candidate": "pilot_cache_v2_candidate/manifest.json",
    }
    for role, value in (("control", control_copy), ("candidate", candidate_copy)):
        extra = value.get("extra_channel")
        if not isinstance(extra, dict):
            raise ValueError("Loop206 checkpoint preprocessing mismatch")
        if (
            extra.get("enabled") is not True
            or extra.get("type") != "loop206_contour_cache"
            or extra.get("require_input_sha256") is not True
        ):
            raise ValueError("Loop206 checkpoint preprocessing mismatch")
        cache_manifest = str(extra.pop("cache_manifest", "")).replace("\\", "/")
        if not cache_manifest.endswith(expected_suffixes[role]):
            raise ValueError("Loop206 checkpoint preprocessing mismatch")
    if control_copy != candidate_copy:
        raise ValueError("Loop206 checkpoint preprocessing mismatch")
    return deepcopy(dict(control))


def _load_torch_endpoint(
    checkpoint_snapshot: ImmutableSnapshot,
    *,
    model_id: str,
    checkpoint_sha256: str,
    device: str,
    seed: int,
) -> tuple[TorchModelEndpoint, dict[str, Any], dict[str, Any]]:
    import torch

    state = torch.load(
        checkpoint_snapshot.open(),
        map_location="cpu",
        weights_only=True,
        mmap=False,
    )
    if not isinstance(state, dict) or not isinstance(state.get("model"), Mapping):
        raise ValueError("Loop206 checkpoint state['model'] is missing")
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Loop206 checkpoint config is missing")
    preprocessing, corruptions = _validate_checkpoint_config(
        config, model_id=model_id, seed=seed
    )
    model = _build_loop206_model()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    model.to(device)
    del state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return (
        TorchModelEndpoint(
            model,
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
        ),
        preprocessing,
        corruptions,
    )


def _validate_pinned_registry(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(PINNED_REGISTRY):
        raise ValueError("Loop206 pinned model registry mismatch")
    if payload.get("schema_version") != PINNED_REGISTRY["schema_version"]:
        raise ValueError("Loop206 pinned model registry mismatch")
    for role in ("control", "candidate"):
        spec = payload.get(role)
        if not isinstance(spec, dict) or spec != PINNED_REGISTRY[role]:
            raise ValueError("Loop206 pinned model registry mismatch")
    for field in ("prior_env", "prior_receipt_env", "prior_receipt_sha256"):
        if payload.get(field) != PINNED_REGISTRY[field]:
            raise ValueError("Loop206 pinned model registry mismatch")
    return payload


def load_model_registry(
    registry_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
) -> LoadedModelRegistry:
    if int(seed) != DEFAULT_SEED:
        raise ValueError("Loop206 pinned model registry seed mismatch")
    registry_snapshot = ImmutableSnapshot.read(registry_path)
    payload = _validate_pinned_registry(json.loads(registry_snapshot.text("ascii")))
    environment = os.environ if environ is None else environ
    resolved: dict[str, tuple[dict[str, Any], ImmutableSnapshot]] = {}
    for role in ("control", "candidate"):
        spec = payload.get(role)
        if not isinstance(spec, dict):
            raise ValueError(f"Loop206 model registry {role} entry mismatch")
        model_id = str(spec.get("model_id", ""))
        env_name = str(spec.get("checkpoint_env", ""))
        expected_hash = str(spec.get("checkpoint_sha256", "")).strip().lower()
        checkpoint_value = str(environment.get(env_name, ""))
        if not model_id or not env_name or len(expected_hash) != 64 or not checkpoint_value:
            raise ValueError(f"Loop206 model registry {role} contract mismatch")
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Loop206 {role} checkpoint is unavailable")
        checkpoint_snapshot = ImmutableSnapshot.read(checkpoint)
        if checkpoint_snapshot.sha256 != expected_hash:
            raise ValueError(f"Loop206 {role} checkpoint hash mismatch")
        resolved[role] = (spec, checkpoint_snapshot)

    if device is None:
        import torch

        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = device
    control, control_preprocessing, control_corruptions = _load_torch_endpoint(
        resolved["control"][1],
        model_id=str(resolved["control"][0]["model_id"]),
        checkpoint_sha256=resolved["control"][1].sha256,
        device=selected_device,
        seed=seed,
    )
    candidate, candidate_preprocessing, candidate_corruptions = _load_torch_endpoint(
        resolved["candidate"][1],
        model_id=str(resolved["candidate"][0]["model_id"]),
        checkpoint_sha256=resolved["candidate"][1].sha256,
        device=selected_device,
        seed=seed,
    )
    preprocessing = _validate_preprocessing_pair(
        control_preprocessing, candidate_preprocessing
    )
    if control_corruptions != candidate_corruptions:
        raise ValueError("Loop206 checkpoint corruption mismatch")

    prior_env = str(payload.get("prior_env", ""))
    receipt_env = str(payload.get("prior_receipt_env", ""))
    prior_value = str(environment.get(prior_env, "")) if prior_env else ""
    receipt_value = str(environment.get(receipt_env, "")) if receipt_env else ""
    if bool(prior_value) != bool(receipt_value):
        raise ValueError("Loop206 prior artifact and receipt must be configured together")
    prior = None
    if prior_value and receipt_value:
        expected_receipt_sha256 = payload.get("prior_receipt_sha256")
        if not isinstance(expected_receipt_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_receipt_sha256
        ):
            raise ValueError("Loop206 prior loading is disabled by the pinned registry")
        prior = load_receipt_authorized_prior(
            prior_value,
            receipt_value,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    return LoadedModelRegistry(
        control,
        candidate,
        preprocessing,
        control_corruptions,
        DEFAULT_SEED,
        prior,
        _REGISTRY_TOKEN,
    )
