"""Fail-closed audit for the files that may be published to GitHub.

``working-tree`` audits tracked files plus non-ignored untracked files.  The
``staged`` mode never reads a path from disk: it audits the exact blobs in the
Git index and rejects index deletes/renames.  This distinction prevents a
clean-looking work tree from masking a dangerous staged blob.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class AuditResult:
    mode: str
    files_scanned: int
    findings: tuple[str, ...]

    @property
    def forbidden(self) -> bool:
        return bool(self.findings)


_FORBIDDEN_SEGMENTS = {
    ".artifacts",
    "checkpoints",
    "data",
    "demo_runtime",
    "logs",
    "masks",
    "raw_receipt",
    "raw_receipts",
    "receipts",
    "runs",
    "tmp",
    "uploads",
    "weights",
}
_SCRATCH_SEGMENT = re.compile(
    r"^(?:\.dg-[^/]+|\.pt[^/]*|\.pytest(?:[-_].*)?|\.task[^/]*|\.p7[^/]*)$",
    re.IGNORECASE,
)
_WEIGHT_SUFFIXES = {
    ".ckpt",
    ".engine",
    ".mmap",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_ARCHIVE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".docker",
    ".img",
    ".ova",
    ".rar",
    ".tar",
    ".tgz",
    ".vhd",
    ".vhdx",
    ".xz",
    ".zip",
}
_PRIVATE_PATH_RE = re.compile(
    r"(?x)(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/][A-Za-z0-9_. -]+(?:[\\/][A-Za-z0-9_. -]+)*|\\\\[A-Za-z0-9_.-]+[\\/][^\s\"'<>]+|(?:/?home|/(?:users|Users|mnt|root|workspaces))[\\/][A-Za-z0-9_.-]+(?:[\\/][^\s\"'<>]*)?|/(?:opt|var)[\\/][A-Za-z0-9_.-]+(?:[\\/][^\s\"'<>]*)?)"
)
_TUNNEL_URL_RE = re.compile(
    r"(?i)https?://[^\s\"'<>]*(?:trycloudflare\.com|cloudflareaccess\.com|ngrok(?:\.io|\.com))[^\s\"'<>]*"
)
_TUNNEL_HOST_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])(?:[A-Za-z0-9-]+\.)+(?:trycloudflare\.com|cloudflareaccess\.com|ngrok\.io|ngrok\.com)\b"
)
_SECRET_RE = re.compile(
    r"(?im)(?:-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})|\b(?:AWS_SECRET_ACCESS_KEY|CLOUDFLARE_API_TOKEN)\s*=\s*[^\s]+)"
)
_ALLOWED_ZIP_SUFFIXES = {".pptx", ".docx"}
_MAX_BLOB_BYTES = 256 * 1024 * 1024
_RECEIPT_NAME_RE = re.compile(
    r"(?:^|[._-])(?:raw[._-]?)?receipts?(?=$|[._-])", re.IGNORECASE
)
_RECEIPT_SCHEMA_RE = re.compile(
    r"(?:^|[._-])receipts?(?=$|[._-])", re.IGNORECASE
)
_PUBLIC_SUMMARY_PATH = (
    "paper/clean_v3_loop206/figures/qualitative_demo_receipts.json"
)
_PUBLIC_SUMMARY_KEYS = {
    "aggregate_mask_bindings_sha256",
    "artifact_role",
    "authorized_sample_count",
    "evidence_class",
    "evidence_registry_sha256",
    "external_runtime_bundle_sha256",
    "panel_caption",
    "provenance_manifest_sha256",
    "release_manifest_sha256",
    "schema_version",
    "source_record_count",
}
_SHA256_TEXT_RE = re.compile(r"[0-9a-f]{64}")


def _run_git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _git_paths(root: Path, *, staged: bool) -> list[tuple[str, bytes, int]]:
    """Return ``(path, blob, mode)`` for an index or work-tree snapshot."""
    if staged:
        records = _run_git(root, "ls-files", "--stage", "-z")
        result: list[tuple[str, bytes, int]] = []
        for record in records.split(b"\0"):
            if not record:
                continue
            metadata, path_bytes = record.split(b"\t", 1)
            mode_text, object_id, _stage = metadata.split()
            path = path_bytes.decode("utf-8", "surrogateescape")
            blob = _run_git(root, "cat-file", "blob", object_id.decode("ascii"))
            result.append((path, blob, int(mode_text, 8)))
        return result

    records = _run_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    result = []
    for raw_path in records.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", "surrogateescape")
        full = root / Path(*PurePosixPath(path).parts)
        try:
            info_before = full.lstat()
            if _is_reparse(info_before):
                result.append((path, b"", stat.S_IFLNK))
                continue
            if not stat.S_ISREG(info_before.st_mode):
                result.append((path, b"", info_before.st_mode))
                continue
            blob = full.read_bytes()
            info_after = full.lstat()
            if _stat_fingerprint(info_before) != _stat_fingerprint(info_after):
                result.append((path, b"__TOCTOU__", info_after.st_mode))
            else:
                result.append((path, blob, info_after.st_mode))
        except OSError:
            result.append((path, b"__UNREADABLE__", 0))
    return result


def _fallback_working_paths(root: Path) -> list[tuple[str, bytes, int]]:
    """Allow unit tests and source bundles without a Git checkout to be audited."""
    result: list[tuple[str, bytes, int]] = []
    for full in sorted(root.rglob("*")):
        if ".git" in full.parts or not full.is_file():
            continue
        path = full.relative_to(root).as_posix()
        try:
            before = full.lstat()
            if _is_reparse(before):
                result.append((path, b"", stat.S_IFLNK))
                continue
            blob = full.read_bytes()
            after = full.lstat()
            result.append((path, b"__TOCTOU__" if _stat_fingerprint(before) != _stat_fingerprint(after) else blob, after.st_mode))
        except OSError:
            result.append((path, b"__UNREADABLE__", 0))
    return result


def _stat_fingerprint(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_size, info.st_mtime_ns, info.st_mode, getattr(info, "st_ino", 0))


def _is_reparse(info: os.stat_result) -> bool:
    attrs = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _staged_changes(root: Path) -> list[str]:
    raw = _run_git(
        root,
        "diff",
        "--cached",
        "--name-status",
        "-z",
        "-C",
        "--find-copies-harder",
    )
    records = raw.split(b"\0")
    findings: list[str] = []
    i = 0
    while i < len(records):
        status = records[i].decode("ascii", "replace") if records[i] else ""
        i += 1
        if not status:
            continue
        if status.startswith("R"):
            old = records[i].decode("utf-8", "replace") if i < len(records) else "?"
            new = records[i + 1].decode("utf-8", "replace") if i + 1 < len(records) else "?"
            i += 2
            findings.append(f"rename staged: {old} -> {new}")
        elif status.startswith("C"):
            old = records[i].decode("utf-8", "replace") if i < len(records) else "?"
            new = records[i + 1].decode("utf-8", "replace") if i + 1 < len(records) else "?"
            i += 2
            findings.append(f"copy staged: {old} -> {new}")
        else:
            path = records[i].decode("utf-8", "replace") if i < len(records) else "?"
            i += 1
            if status.startswith("D"):
                findings.append(f"delete staged: {path}")
    return findings


def _path_findings(path: str) -> list[str]:
    normalized = path.replace("\\", "/")
    parts = [part.lower() for part in PurePosixPath(normalized).parts]
    findings: list[str] = []
    for part in parts:
        safe_evidence_data = (
            part == "data"
            and normalized.lower().startswith("demo/data/")
            and normalized.lower().endswith("evidence_registry.json")
        )
        if part in _FORBIDDEN_SEGMENTS and not safe_evidence_data:
            findings.append(f"forbidden path segment: {path}")
        if _SCRATCH_SEGMENT.match(part):
            findings.append(f"scratch path: {path}")
    lower = normalized.lower()
    suffix = Path(lower).suffix
    suffixes = set(Path(lower).suffixes)
    if suffixes & _WEIGHT_SUFFIXES:
        findings.append(f"model weight/artifact: {path}")
    if suffixes & _ARCHIVE_SUFFIXES:
        findings.append(f"binary archive: {path}")
    if suffix == ".log" or Path(lower).name.startswith("cloudflared"):
        findings.append(f"runtime log: {path}")
    if _RECEIPT_NAME_RE.search(Path(lower).name) or "raw_receipt" in lower:
        findings.append(f"raw receipt: {path}")
    return findings


def _is_public_aggregate_summary(path: str, blob: bytes) -> bool:
    if path.replace("\\", "/").lower() != _PUBLIC_SUMMARY_PATH:
        return False
    try:
        payload = json.loads(blob.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != _PUBLIC_SUMMARY_KEYS:
        return False
    digest_keys = {
        "aggregate_mask_bindings_sha256",
        "evidence_registry_sha256",
        "external_runtime_bundle_sha256",
        "provenance_manifest_sha256",
        "release_manifest_sha256",
    }
    return (
        payload.get("schema_version") == "loop206.qualitative_public_summary.v1"
        and payload.get("artifact_role") == "derived_public_aggregate_provenance"
        and payload.get("evidence_class")
        == "train_screen / exact_fixed_cache / historical_cache_provenance_drift"
        and payload.get("panel_caption")
        == "illustrative; not protected-test evidence"
        and payload.get("authorized_sample_count") == 3
        and payload.get("source_record_count") == 3
        and all(
            isinstance(payload.get(key), str)
            and _SHA256_TEXT_RE.fullmatch(payload[key])
            for key in digest_keys
        )
    )


def _json_receipt_findings(path: str, text: str) -> list[str]:
    if Path(path).suffix.lower() != ".json":
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    findings: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key).lower().replace("-", "_")
                if normalized_key in {"receipts", "raw_receipts"} and isinstance(
                    child, list
                ):
                    findings.append(f"raw receipt array: {path}")
                if (
                    normalized_key in {"schema", "schema_id", "schema_version"}
                    or normalized_key.endswith("_schema_id")
                    or normalized_key.endswith("_schema_version")
                ) and isinstance(child, str) and _RECEIPT_SCHEMA_RE.search(child):
                    findings.append(f"raw receipt schema: {path}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return findings


def _content_findings(path: str, blob: bytes) -> list[str]:
    findings: list[str] = []
    if len(blob) > _MAX_BLOB_BYTES:
        return [f"oversized blob: {path}"]
    if blob == b"__TOCTOU__":
        return [f"TOCTOU detected: {path}"]
    if blob == b"__UNREADABLE__":
        return [f"unreadable file: {path}"]
    suffix = Path(path.lower()).suffix
    if blob.startswith(b"PK\x03\x04") and suffix not in _ALLOWED_ZIP_SUFFIXES:
        findings.append(f"binary archive signature: {path}")
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return findings
    if _TUNNEL_URL_RE.search(text) or _TUNNEL_HOST_RE.search(text):
        findings.append(f"tunnel URL: {path}")
    if _PRIVATE_PATH_RE.search(text):
        findings.append(f"private absolute path: {path}")
    if _SECRET_RE.search(text):
        findings.append(f"secret-like material: {path}")
    findings.extend(_json_receipt_findings(path, text))
    if Path(path).name.lower() == "result_manifest.json":
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            findings.append(f"malformed result manifest: {path}")
        else:
            if not isinstance(payload, dict) or payload.get("status") != "pending/unverified" or payload.get("p1_status") != "not_promoted" or payload.get("metrics") != [] or payload.get("completed_jobs") != 0 or payload.get("required_jobs") != 6:
                findings.append(f"result manifest is not blocked/pending: {path}")
    return findings


def audit(repo_root: Path | str, *, mode: str = "working-tree") -> AuditResult:
    root = Path(repo_root).resolve()
    if mode not in {"working-tree", "staged"}:
        raise ValueError("mode must be working-tree or staged")
    findings: list[str] = []
    index_before: bytes | None = None
    try:
        if mode == "staged":
            index_before = _run_git(root, "ls-files", "--stage", "-z")
        entries = _git_paths(root, staged=mode == "staged")
    except (OSError, subprocess.CalledProcessError) as exc:
        if mode == "working-tree":
            entries = _fallback_working_paths(root)
        else:
            return AuditResult(mode, 0, (f"git snapshot unavailable: {exc}",))
    if mode == "staged":
        findings.extend(_staged_changes(root))
    for path, blob, mode_bits in entries:
        path_findings = _path_findings(path)
        if _is_public_aggregate_summary(path, blob):
            path_findings = [
                finding
                for finding in path_findings
                if finding != f"raw receipt: {path}"
            ]
        findings.extend(path_findings)
        if stat.S_ISLNK(mode_bits) or mode_bits == 0o120000:
            findings.append(f"symlink/reparse entry: {path}")
            continue
        if not stat.S_ISREG(mode_bits):
            findings.append(f"non-regular entry: {path}")
            continue
        findings.extend(_content_findings(path, blob))
    if mode == "staged":
        # A staged index change during the audit invalidates the snapshot.
        try:
            index_after = _run_git(root, "ls-files", "--stage", "-z")
            if index_before is not None and index_after != index_before:
                findings.append("index changed during audit (TOCTOU)")
        except (OSError, subprocess.CalledProcessError) as exc:
            findings.append(f"index verification failed: {exc}")
    return AuditResult(mode, len(entries), tuple(dict.fromkeys(findings)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("working-tree", "staged"), default="working-tree")
    args = parser.parse_args(argv)
    result = audit(args.repo_root, mode=args.mode)
    if result.forbidden:
        for finding in result.findings:
            print(f"FAIL: {finding}", file=sys.stderr)
        return 1
    print(f"PASS: {result.mode} public surface ({result.files_scanned} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
