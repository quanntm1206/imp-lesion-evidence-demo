from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import json
import re
import socket
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    SidecarResult,
    mask_sha256,
    rgb_sha256,
)
from lesion_robustness.demo.dual_live_service import DualLiveService
from lesion_robustness.demo.nnunet_determinism import missing_runtime_prerequisites
from tests.demo.support.one_shot_nnunet_client import OneShotTimeoutClient


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _ImpResult:
    control_mask: np.ndarray
    control_overlay: np.ndarray
    control_latency_ms: float
    device: str = "cuda:0"
    control_model_id: str = "L206-control-s206"
    control_checkpoint_sha256: str = (
        "be606b0a0940839b019ea60117dda4b27f9b8f04d54306b5b676f2c29516fcef"
    )


class _LocalImp:
    def preview_control(self, image: np.ndarray) -> _ImpResult:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return _ImpResult(mask, image.copy(), 1.0)


class _RealLocalClient:
    def __init__(self) -> None:
        self.identity = object()
        self.calls = 0

    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult:
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return SidecarResult(
            request_id=request_id,
            input_sha256=rgb_sha256(image),
            mask=mask,
            mask_sha256=mask_sha256(mask),
            model_id=MODEL_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            latency_ms=1.0,
            execution="live",
            protocol=PROTOCOL_ID,
        )


def test_local_only_injected_client_fails_closed_then_recovers_without_restart() -> None:
    real_local_client = _RealLocalClient()
    sidecar_identity = real_local_client.identity
    service = DualLiveService(
        _LocalImp(), OneShotTimeoutClient(real_local_client)
    )
    image = np.full((8, 8, 3), 127, dtype=np.uint8)

    failed = service.run(image)
    assert failed.imp is not None
    assert failed.nnunet is not None and failed.nnunet.status == "failed"
    assert failed.receipt_eligible is False
    assert real_local_client.calls == 0

    recovered = service.run(image)
    assert recovered.imp is not None
    assert recovered.nnunet is not None and recovered.nnunet.status == "completed"
    assert recovered.receipt_eligible is True
    assert real_local_client.calls == 1
    assert real_local_client.identity is sidecar_identity


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _python_surface(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="ascii"), filename=str(path))
    names: set[str] = set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            function = node.func
            attribute = function.attr if isinstance(function, ast.Attribute) else ""
            if attribute == "add_argument":
                names.update(
                    arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                )
            if attribute in {
                "get",
                "getenv",
                "route",
                "add_route",
                "add_api_route",
            } and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
            names.update(
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg in {"api_name", "route", "endpoint", "test_journal"}
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            )
            names.update(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg == "test_journal"
            )
    return names, imports


def _forbidden_fault_surfaces(names: set[str]) -> set[str]:
    forbidden_fragments = ("fault", "inject", "timeoutonce", "testjournal")
    forbidden_exact = {"oneshotnnunetclient"}
    return {
        name
        for name in names
        if (
            _normalized(name) in forbidden_exact
            or any(
                fragment in _normalized(name) for fragment in forbidden_fragments
            )
        )
    }


def test_semantic_surface_extracts_all_fault_eligible_names(tmp_path: Path) -> None:
    source = tmp_path / "surface.py"
    source.write_text(
        """\
import os
os.getenv("INJECT_FAILURE")
app.route("/fault/trigger")
register(api_name="safe_api", endpoint="FAULT_MODE", test_journal="safe-journal")
""",
        encoding="ascii",
    )

    names, _imports = _python_surface(source)

    assert {
        "INJECT_FAILURE",
        "/fault/trigger",
        "safe_api",
        "FAULT_MODE",
        "test_journal",
        "safe-journal",
    }.issubset(names)


def test_semantic_surface_forbids_containment_without_banning_benign_names() -> None:
    forbidden = {
        "INJECT_FAILURE",
        "FAULT_MODE",
        "/fault/trigger",
        "timeout-once-api",
        "test_journal",
        "test_journal_mode",
        "TeSt-JoUrNaL-hook",
    }
    allowed = {
        "dual_live_compare",
        "IMP_LOOP206_DATA_ROOT",
        "/health",
        "request_timeout_seconds",
        "inference_endpoint",
        "runtime_journal",
        "audit-journal",
        "safe-journal",
    }

    assert _forbidden_fault_surfaces(forbidden) == forbidden
    assert _forbidden_fault_surfaces(allowed) == set()


def test_fault_support_is_absent_from_semantic_production_surfaces() -> None:
    names: set[str] = set()
    imports: set[str] = set()
    for path in (ROOT / "src/lesion_robustness/demo").rglob("*.py"):
        path_names, path_imports = _python_surface(path)
        names.update(path_names)
        imports.update(path_imports)
    for path in (
        ROOT / "scripts/demo/run_demo.ps1",
        ROOT / "scripts/demo/run_sidecar.ps1",
        ROOT / "scripts/demo/run_tunnel.ps1",
    ):
        text = path.read_text(encoding="ascii")
        names.update(re.findall(r"\$env:([A-Za-z0-9_]+)", text, flags=re.IGNORECASE))
        names.update(re.findall(r"^\s*\[(?:switch|string)\]\$([A-Za-z0-9_]+)", text, flags=re.MULTILINE | re.IGNORECASE))
    config = (ROOT / "configs/demo/loop206_live.yaml").read_text(encoding="ascii")
    names.update(re.findall(r"^\s*([A-Za-z0-9_-]+)\s*:", config, flags=re.MULTILINE))

    assert _forbidden_fault_surfaces(names) == set()
    assert all(not module.casefold().startswith("tests.") for module in imports)


def test_one_shot_timeout_is_exactly_once_under_concurrency() -> None:
    real_local_client = _RealLocalClient()
    client = OneShotTimeoutClient(real_local_client)
    image = np.full((8, 8, 3), 127, dtype=np.uint8)

    def call(index: int) -> str:
        try:
            client.predict(f"{index:032x}", image)
        except Exception as exc:
            return type(exc).__name__ + ":" + str(exc)
        return "completed"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(call, range(8)))

    assert outcomes.count("SidecarUnavailable:timeout") == 1
    assert outcomes.count("completed") == 7
    assert real_local_client.calls == 7


def test_runtime_prerequisite_inventory_is_fail_closed_before_ports(tmp_path: Path) -> None:
    missing = missing_runtime_prerequisites(
        tmp_path,
        environ={},
        tool_finder=lambda _name: None,
    )
    assert missing == (
        "python",
        "docker_cuda",
        "sidecar_bundle",
        "imp_control_checkpoint",
        "imp_candidate_checkpoint",
        "dataset_index",
        "candidate_cache_manifest",
        "zero_control_cache_manifest",
        "release_manifest",
        "evidence_registry",
        "clean_v3_manifest",
    )


def test_reported_blocked_packet_is_canonical_and_has_no_downstream_claims() -> None:
    path = (
        ROOT
        / "demo_runtime/acceptance/imp.dual_live.e2e.v1"
        / "20260722T233947051Z/acceptance.json"
    )
    raw = path.read_bytes()
    packet = json.loads(raw.decode("ascii"))
    canonical = (
        json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    assert raw == canonical
    assert packet["status"] == "blocked_missing_prerequisite"
    assert packet["missing_artifact_classes"] == [
        "imp_control_checkpoint",
        "imp_candidate_checkpoint",
        "clean_v3_manifest",
    ]
    assert packet["claim_status"] == "unpromoted"
    assert not any(key.endswith("_sha256") for key in packet)
    assert {
        "browser",
        "cloudflare",
        "tunnel",
        "cleanup",
        "screenshots",
        "ports_bound",
    }.isdisjoint(packet)


def test_blocked_runtime_ports_are_closed() -> None:
    for port in (7860, 7861, 7862):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.1)
            assert connection.connect_ex(("127.0.0.1", port)) != 0
