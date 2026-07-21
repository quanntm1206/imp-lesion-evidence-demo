from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED = {
    "checkpoint_final.pth": ("checkpoint_sha256", "checkpoint"),
    "nnUNetPlans.json": ("plans_sha256", "plans"),
    "dataset_fingerprint.json": ("fingerprint_sha256", "fingerprint"),
}
METADATA = (
    "dataset.json",
    "plans.json",
    "runtime_identity.json",
    "requirements.lock",
)


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report["provenance"]
    observed: dict[str, str] = {}
    for filename, (key, label) in REQUIRED.items():
        value = sha256_file(bundle / filename)
        if value != str(provenance[key]):
            raise ValueError(f"{label} hash mismatch")
        observed[key] = value

    metadata: dict[str, dict[str, Any]] = {}
    for filename in METADATA:
        artifact = bundle / filename
        if not artifact.is_file():
            raise FileNotFoundError(f"required Loop192 metadata missing: {filename}")
        metadata[filename] = {
            "sha256": sha256_file(artifact),
            "size": artifact.stat().st_size,
        }

    return {
        "schema_version": "loop192.recovery.receipt.v1",
        "model_id": str(report["candidate_id"]),
        **observed,
        "metadata": metadata,
        "source_vhd_unchanged": True,
    }
