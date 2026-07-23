"""Runtime identity compatibility projection; canonical values live in release JSON."""

from __future__ import annotations

from lesion_robustness.release_manifest import runtime_projection


def _runtime() -> dict[str, object]:
    return runtime_projection()


def runtime_identities() -> tuple[dict[str, object], dict[str, object]]:
    runtime = _runtime()
    return dict(runtime["imp"]), dict(runtime["nnunet"])


RUNTIME = _runtime()
IMP = dict(RUNTIME["imp"])
NNUNET = dict(RUNTIME["nnunet"])
MODEL_ID = str(NNUNET["model_id"])
CHECKPOINT_SHA256 = str(NNUNET["checkpoint_sha256"])
PROTOCOL_ID = str(dict(NNUNET["runtime"])["protocol"])
