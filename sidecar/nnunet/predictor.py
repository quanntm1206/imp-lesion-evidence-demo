"""Persistent, CUDA-only adapter for the recovered Loop192 nnU-Net model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import hmac
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image

from lesion_robustness.demo.dual_live_protocol import MODEL_ID, validate_binary_mask, validate_rgb


MANIFEST_SCHEMA = "imp.nnunet.model-manifest.v1"
REQUIRED_ARTIFACTS = frozenset(
    {
        "checkpoint_final.pth",
        "plans.json",
        "dataset.json",
        "dataset_fingerprint.json",
    }
)
EXPECTED_RUNTIME_VERSION = "2.8.1"
EXPECTED_RUNTIME_COMMIT = "3e9fdc5fec7c8164f8fc2c6263af8be73278130e"
NATURAL_IMAGE_SPACING = (999.0, 1.0, 1.0)


class ArtifactDriftError(RuntimeError):
    """Raised without artifact details when trusted hashes no longer match."""

    public_code = "artifact_drift"

    def __init__(self) -> None:
        super().__init__(self.public_code)


class RuntimeConfigurationError(RuntimeError):
    """Raised without local details when the pinned CUDA runtime is unavailable."""

    public_code = "runtime_unavailable"

    def __init__(self) -> None:
        super().__init__(self.public_code)


class InferenceOOMError(RuntimeError):
    """Sanitized CUDA out-of-memory signal for the HTTP boundary."""

    public_code = "oom"

    def __init__(self) -> None:
        super().__init__(self.public_code)


RuntimeLoader = Callable[[], tuple[Any, type[Any], str]]


def _load_runtime() -> tuple[Any, type[Any], str]:
    version = importlib.metadata.version("nnunetv2")
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    return torch, nnUNetPredictor, version


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "model_id",
            "runtime",
            "input",
            "artifacts",
        }:
            raise ValueError
        runtime = raw["runtime"]
        input_contract = raw["input"]
        artifacts = raw["artifacts"]
        if raw["schema_version"] != MANIFEST_SCHEMA or raw["model_id"] != MODEL_ID:
            raise ValueError
        if not isinstance(runtime, Mapping) or dict(runtime) != {
            "distribution": "nnunetv2",
            "version": EXPECTED_RUNTIME_VERSION,
            "recovered_git_commit": EXPECTED_RUNTIME_COMMIT,
            "environment_status": "reconstructed",
        }:
            raise ValueError
        if not isinstance(input_contract, Mapping) or dict(input_contract) != {
            "layout": "CZYX",
            "channels": 3,
            "spacing": list(NATURAL_IMAGE_SPACING),
        }:
            raise ValueError
        if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_ARTIFACTS:
            raise ValueError
        for entry in artifacts.values():
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"sha256", "size"}
                or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in entry["sha256"])
                or isinstance(entry["size"], bool)
                or not isinstance(entry["size"], int)
                or entry["size"] < 1
            ):
                raise ValueError
        return raw
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactDriftError() from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(bundle: Path, manifest: Mapping[str, Any]) -> None:
    try:
        for filename, expected in manifest["artifacts"].items():
            artifact = bundle / filename
            before = artifact.stat()
            observed_hash = _sha256(artifact)
            after = artifact.stat()
            stable = (
                before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and before.st_ctime_ns == after.st_ctime_ns
            )
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or not stable
                or after.st_size != expected["size"]
                or not hmac.compare_digest(observed_hash, expected["sha256"])
            ):
                raise ArtifactDriftError()
    except (OSError, KeyError, TypeError) as exc:
        raise ArtifactDriftError() from exc


def _assert_runtime_api(predictor_class: type[Any]) -> None:
    expected = {
        "initialize_from_trained_model_folder": (
            "self",
            "model_training_output_dir",
            "use_folds",
            "checkpoint_name",
        ),
        "predict_single_npy_array": (
            "self",
            "input_image",
            "image_properties",
            "segmentation_previous_stage",
            "output_file_truncated",
            "save_or_return_probabilities",
        ),
    }
    try:
        for method_name, parameters in expected.items():
            method = getattr(predictor_class, method_name)
            if tuple(inspect.signature(method).parameters) != parameters:
                raise RuntimeConfigurationError()
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeConfigurationError() from exc


def _link_artifact(source: Path, destination: Path) -> None:
    try:
        os.symlink(source, destination, target_is_directory=False)
    except OSError:
        try:
            os.link(source, destination)
        except OSError as exc:
            raise RuntimeConfigurationError() from exc


def _create_model_layout(bundle: Path) -> tempfile.TemporaryDirectory[str]:
    temp_parent = bundle.parent if os.name == "nt" else None
    owned = tempfile.TemporaryDirectory(prefix="imp-nnunet-", dir=temp_parent)
    try:
        root = Path(owned.name)
        fold = root / "fold_all"
        fold.mkdir()
        for filename in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
            _link_artifact(bundle / filename, root / filename)
        _link_artifact(bundle / "checkpoint_final.pth", fold / "checkpoint_final.pth")
        return owned
    except BaseException:
        owned.cleanup()
        raise


class Loop192Predictor:
    """Load Loop192 once, then serve one validated RGB image per call."""

    model_id = MODEL_ID
    device = "cuda:0"

    def __init__(
        self,
        model_bundle: str | Path,
        model_manifest: str | Path,
        *,
        runtime_loader: RuntimeLoader = _load_runtime,
    ) -> None:
        self.ready = False
        bundle = Path(model_bundle).resolve(strict=True)
        manifest = _load_manifest(Path(model_manifest).resolve(strict=True))
        _verify_artifacts(bundle, manifest)
        self.checkpoint_sha256 = str(
            manifest["artifacts"]["checkpoint_final.pth"]["sha256"]
        )

        try:
            torch, predictor_class, version = runtime_loader()
        except ArtifactDriftError:
            raise
        except BaseException as exc:
            raise RuntimeConfigurationError() from exc
        if version != EXPECTED_RUNTIME_VERSION or not torch.cuda.is_available():
            raise RuntimeConfigurationError()
        _assert_runtime_api(predictor_class)

        self._torch = torch
        self._model_layout = _create_model_layout(bundle)
        try:
            self._predictor = predictor_class(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device("cuda", 0),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            self._predictor.initialize_from_trained_model_folder(
                self._model_layout.name,
                use_folds=("all",),
                checkpoint_name="checkpoint_final.pth",
            )
        except BaseException:
            self._model_layout.cleanup()
            raise
        self.ready = True

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        rgb = validate_rgb(image)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        nnunet_input = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[:, None, :, :], dtype=np.float32
        )
        prediction: np.ndarray | None = None
        try:
            self._torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = self._predictor.predict_single_npy_array(
                nnunet_input,
                {"spacing": NATURAL_IMAGE_SPACING},
                segmentation_previous_stage=None,
                output_file_truncated=None,
                save_or_return_probabilities=False,
            )
            self._torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            mask = self._normalize_mask(prediction, height, width)
            return mask, latency_ms
        except BaseException as exc:
            oom_type = getattr(self._torch.cuda, "OutOfMemoryError", None)
            if isinstance(oom_type, type) and isinstance(exc, oom_type):
                raise InferenceOOMError() from exc
            raise
        finally:
            del nnunet_input
            if prediction is not None:
                del prediction
            self._torch.cuda.empty_cache()

    @staticmethod
    def _normalize_mask(prediction: np.ndarray, height: int, width: int) -> np.ndarray:
        value = np.asarray(prediction)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
            raise ValueError("invalid_prediction")
        binary = value.astype(np.uint8, copy=False)
        if binary.shape != (height, width):
            binary = np.asarray(
                Image.fromarray(binary, mode="L").resize(
                    (width, height), resample=Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
        return validate_binary_mask(binary)

