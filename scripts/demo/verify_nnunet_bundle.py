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
MODEL_ID = "L192-nnUNet-v2-raw-100ep"
VHD_PROOF_FIELDS = ("length", "creation_time_utc", "last_write_time_utc")


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_local_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in ("\\", "/")
    ) or value.startswith("\\\\") or normalized.startswith(("/home/", "/mnt/"))


def _verify_source_vhd_proof(report: Mapping[str, Any]) -> None:
    try:
        proof = report["source_vhd_proof"]
        before = proof["before"]
        after = proof["after"]
    except (KeyError, TypeError) as error:
        raise ValueError("source VHD unchanged proof required") from error
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError("source VHD unchanged proof required")
    for field in VHD_PROOF_FIELDS:
        try:
            before_value = before[field]
            after_value = after[field]
        except KeyError as error:
            raise ValueError("source VHD unchanged proof required") from error
        if before_value != after_value:
            raise ValueError("source VHD changed after recovery")
        if field == "length":
            if not isinstance(before_value, int) or before_value < 0:
                raise ValueError("source VHD unchanged proof required")
        elif not isinstance(before_value, str) or not before_value:
            raise ValueError("source VHD unchanged proof required")


def verify_bundle(bundle: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report["provenance"]
    candidate_id = report["candidate_id"]
    if not isinstance(candidate_id, str):
        raise ValueError("candidate_id must be a string")
    if _contains_local_path(candidate_id):
        raise ValueError("candidate_id must not contain a local path")
    if candidate_id != MODEL_ID:
        raise ValueError("candidate_id does not match Loop192")
    _verify_source_vhd_proof(report)
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
        "model_id": candidate_id,
        **observed,
        "metadata": metadata,
        "source_vhd_unchanged": True,
    }
