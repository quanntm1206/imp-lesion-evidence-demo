from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
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
PINNED_HASHES = MappingProxyType(
    {
        "checkpoint_sha256": "3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2",
        "plans_sha256": "b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7",
        "fingerprint_sha256": "931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c",
    }
)
VHD_PROOF_FIELDS = ("length", "creation_time_utc", "last_write_time_utc")
RECEIPT_CONTEXT = MappingProxyType(
    {
        "recovery_backend": "container-readonly-7zip",
        "parser_warning": "Headers Error",
        "runtime_status": "reconstructed_required",
    }
)


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
        expected = PINNED_HASHES[key]
        if str(provenance[key]) != expected:
            raise ValueError(f"{label} provenance does not match Loop192 pin")
        value = sha256_file(bundle / filename)
        if value != expected:
            raise ValueError(f"{label} hash does not match Loop192 pin")
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


def _is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _resolve_regular_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} must be a regular file") from error
    if _is_reparse(info):
        raise ValueError(f"{label} must not be a reparse point")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _resolve_receipt_path(bundle: Path, receipt: Path) -> Path:
    try:
        parent = receipt.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("receipt parent must equal resolved bundle") from error
    if parent != bundle:
        raise ValueError("receipt parent must equal resolved bundle")
    if receipt.name != "recovery_receipt.json":
        raise ValueError("receipt basename must be exactly recovery_receipt.json")
    resolved = parent / receipt.name
    try:
        info = resolved.lstat()
    except FileNotFoundError:
        return resolved
    if _is_reparse(info):
        raise ValueError("receipt must not be a reparse point")
    raise ValueError("receipt must not already exist")


def _write_receipt_atomic(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("receipt must not already exist") from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the recovered Loop192 bundle",
        allow_abbrev=False,
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--report", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)

    bundle = args.bundle.resolve(strict=True)
    if not bundle.is_dir():
        raise ValueError("bundle must be a directory")
    receipt_path = _resolve_receipt_path(bundle, args.receipt)
    if args.report == "-":
        report_text = sys.stdin.read()
    else:
        report_path = _resolve_regular_file(Path(args.report), label="report")
        if report_path == receipt_path:
            raise ValueError("report and receipt must differ")
        report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    if not isinstance(report, Mapping):
        raise ValueError("report must contain a JSON object")

    receipt = verify_bundle(bundle, report)
    for key, expected in RECEIPT_CONTEXT.items():
        if key in report:
            if report[key] != expected:
                raise ValueError(f"{key} does not match the trusted receipt context")
            receipt[key] = expected
    _write_receipt_atomic(receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
