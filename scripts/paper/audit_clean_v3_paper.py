"""Fail closed when the evidence-bound paper or demo drifts from its registry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from lesion_robustness.evidence_registry import validate_registry


_CITATION = re.compile(
    r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*){0,2}\{([^}]+)\}"
)
_BIB_KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z_0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_RESULT_TERMS = re.compile(
    r"robust[- ]?(?:dice|iou)|boundary[- ]?f1|(?:^|\W)(?:dice|iou|precision|recall|hd95|assd)(?:\W|$)|"
    r"point estimate|confidence interval|candidate-minus-control|\bdelta\b",
    re.IGNORECASE,
)
_UNFINISHED = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|\?\?")
_CLAIM_TERMS = re.compile(
    r"state[ -]of[ -]the[ -]art|\bsota\b|statistical(?:ly)?[ -]superior(?:ity)?|"
    r"clinical[- ]grade|clinical validation|clinical system|clinical use|diagnostic(?: claim| system| use)?|"
    r"protected[- ]test (?:result|evidence|claim)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|unavailable|sealed|prevent(?:s|ed)?|"
    r"does not|do not|did not|cannot|is not|are not|has not|have not|rather than)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditResult:
    errors: tuple[str, ...]
    registry_sha256: str | None

    @property
    def passed(self) -> bool:
        return not self.errors

    def receipt(self, paper: Path) -> dict[str, Any]:
        return {
            "errors": list(self.errors),
            "paper": paper.name,
            "passed": self.passed,
            "registry_sha256": self.registry_sha256,
            "schema_version": "imp.paper_audit.v1",
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, errors: list[str], *, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding)
    except (OSError, UnicodeError):
        errors.append(f"unreadable input: {path.name}")
        return ""


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _walk_numbers(value: object) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_numbers(item)


def _supported_numbers(registry: dict[str, Any]) -> set[float]:
    values = set(_walk_numbers(registry))
    values.update(abs(value) for value in tuple(values))
    observations = registry.get("observations", [])
    metric_fields = (
        "robust_dice",
        "robust_iou",
        "robust_precision",
        "robust_recall",
        "robust_bf1",
    )
    if isinstance(observations, list):
        for left in observations:
            if not isinstance(left, dict):
                continue
            for right in observations:
                if not isinstance(right, dict):
                    continue
                for field in metric_fields:
                    left_value = left.get(field)
                    right_value = right.get(field)
                    if isinstance(left_value, (int, float)) and isinstance(
                        right_value, (int, float)
                    ):
                        values.add(abs(float(left_value) - float(right_value)))
    return values


def _number_is_supported(token: str, supported: set[float]) -> bool:
    value = float(token.replace(",", ""))
    if value in {95.0}:  # Confidence-level notation, not an empirical result.
        return True
    tolerance = max(0.00006, abs(value) * 1e-10)
    return any(abs(value - candidate) <= tolerance for candidate in supported)


def _sentences(text: str) -> Iterable[str]:
    return re.split(r"(?<=[.!?])\s+|\n+", text)


def _claim_is_negated(sentence: str) -> bool:
    return bool(_NEGATION.search(sentence))


def _check_claims(path: Path, text: str, root: Path, errors: list[str]) -> None:
    for sentence in _sentences(text):
        if _CLAIM_TERMS.search(sentence) and not _claim_is_negated(sentence):
            errors.append(f"affirmative protected claim: {_relative(path, root)}")
            return


def _check_citations(
    tex_files: Iterable[Path], bib_text: str, root: Path, errors: list[str]
) -> None:
    defined = set(_BIB_KEY.findall(bib_text))
    for path in tex_files:
        for keys in _CITATION.findall(_read(path, errors)):
            for key in (value.strip() for value in keys.split(",")):
                if key and key not in defined:
                    errors.append(
                        f"undefined citation key: {key} in {_relative(path, root)}"
                    )


def _check_manifest(
    paper: Path,
    registry: dict[str, Any],
    root: Path,
    errors: list[str],
) -> dict[str, Any] | None:
    path = paper / "artifact_manifest.json"
    if not path.is_file():
        errors.append("missing evidence mapping")
        return None
    try:
        manifest = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("invalid artifact manifest")
        return None
    if manifest.get("evidence_registry_sha256") != registry.get("registry_sha256"):
        errors.append("missing evidence mapping")
    for category, label in (("figures", "figure"), ("tables", "table")):
        entries = manifest.get(category, {})
        if not isinstance(entries, dict):
            errors.append(f"invalid {category} manifest")
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                errors.append(f"invalid {label} manifest entry")
                continue
            _check_hashed_artifact(paper, entry, "path", "sha256", label, errors)
            _check_hashed_artifact(
                paper,
                entry,
                "generation_source_path",
                "generation_source_sha256",
                "source",
                errors,
            )
            _check_hashed_artifact(
                paper,
                entry,
                "capture_source_path",
                "capture_source_sha256",
                "source",
                errors,
            )
            _check_hashed_artifact(
                paper,
                entry,
                "receipt_path",
                "receipt_sha256",
                "source",
                errors,
            )
    return manifest


def _check_hashed_artifact(
    paper: Path,
    entry: dict[str, Any],
    path_key: str,
    hash_key: str,
    label: str,
    errors: list[str],
) -> None:
    source = entry.get(path_key)
    expected = entry.get(hash_key)
    if source is None and expected is None:
        return
    if not isinstance(source, str) or not isinstance(expected, str):
        errors.append(f"missing {label} hash")
        return
    artifact = paper / source
    if not artifact.is_file():
        errors.append(f"missing {label} hash")
    elif _sha256(artifact) != expected:
        errors.append(f"{label} hash drift")


def _check_source_hashes(
    registry: dict[str, Any], root: Path, errors: list[str]
) -> None:
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("missing evidence mapping")
        return
    for source in sources:
        if not isinstance(source, dict):
            errors.append("invalid registry source")
            continue
        relative = source.get("path")
        expected = source.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append("invalid registry source")
            continue
        candidate = root / relative
        if not candidate.is_file() or _sha256(candidate) != expected:
            errors.append(f"source hash drift: {source.get('source_id', 'unknown')}")


def _check_loop170_labels(tex_files: Iterable[Path], root: Path, errors: list[str]) -> None:
    for path in tex_files:
        text = _read(path, errors)
        if re.search(r"loop170", text, re.IGNORECASE) and _NUMBER.search(text):
            label = "legacy_patient_contaminated"
            if label not in text.replace("\\_", "_"):
                errors.append(f"unlabeled Loop170 values: {_relative(path, root)}")


def _check_result_numbers(
    paper: Path, registry: dict[str, Any], root: Path, errors: list[str]
) -> None:
    supported = _supported_numbers(registry)
    files = [paper / "main.tex", *sorted((paper / "tables").glob("*.tex"))]
    results = paper / "sections" / "06_results.tex"
    if results.is_file():
        files.append(results)
    for path in files:
        if not path.is_file():
            continue
        for sentence in _sentences(_read(path, errors)):
            if not _RESULT_TERMS.search(sentence):
                continue
            for match in _NUMBER.finditer(sentence):
                if not _number_is_supported(match.group(), supported):
                    errors.append(
                        f"unsupported numeric result: {match.group()} in {_relative(path, root)}"
                    )


def _check_demo_receipts(
    paper: Path, registry: dict[str, Any], errors: list[str]
) -> None:
    receipt = paper / "figures" / "qualitative_demo_receipts.json"
    if not receipt.is_file():
        return
    try:
        payload = json.loads(receipt.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("invalid demo receipt bundle")
        return
    if payload.get("evidence_registry_sha256") != registry.get("registry_sha256"):
        errors.append("demo registry hash drift")
    entries = payload.get("receipts")
    if not isinstance(entries, list):
        errors.append("invalid demo receipt bundle")
        return
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("invalid demo receipt bundle")
            continue
        if "metrics" not in entry:
            continue
        authorization = entry.get("display_authorization")
        if not isinstance(authorization, dict) or authorization.get(
            "mask_variant"
        ) != "challenge_ground_truth":
            errors.append("hidden no-GT metrics")


def audit_paper(paper: str | Path, registry: str | Path) -> AuditResult:
    """Audit paper inputs without emitting absolute filesystem paths."""
    paper_path = Path(paper).resolve()
    registry_path = Path(registry).resolve()
    root = paper_path.parents[1] if len(paper_path.parents) > 1 else paper_path.parent
    errors: list[str] = []
    registry_payload: dict[str, Any] | None = None
    try:
        loaded = json.loads(registry_path.read_text(encoding="ascii"))
        if not isinstance(loaded, dict):
            raise ValueError("registry must be an object")
        validate_registry(loaded)
        registry_payload = loaded
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        errors.append("invalid evidence registry")

    tex_files = sorted(paper_path.rglob("*.tex")) if paper_path.is_dir() else []
    bib_files = sorted(paper_path.rglob("*.bib")) if paper_path.is_dir() else []
    readme = root / "README.md"
    texts = [(path, _read(path, errors)) for path in [*tex_files, *bib_files]]
    if readme.is_file():
        texts.append((readme, _read(readme, errors)))
    for path, text in texts:
        if _UNFINISHED.search(text):
            errors.append(f"unfinished marker: {_relative(path, root)}")
        if path.suffix.lower() == ".tex" or path == readme:
            _check_claims(path, text, root, errors)
    bib_text = "\n".join(text for path, text in texts if path.suffix.lower() == ".bib")
    _check_citations(tex_files, bib_text, root, errors)

    if registry_payload is not None:
        _check_manifest(paper_path, registry_payload, root, errors)
        _check_source_hashes(registry_payload, root, errors)
        _check_loop170_labels(tex_files, root, errors)
        _check_result_numbers(paper_path, registry_payload, root, errors)
        _check_demo_receipts(paper_path, registry_payload, errors)
        registry_sha256 = str(registry_payload.get("registry_sha256"))
    else:
        registry_sha256 = None
    return AuditResult(tuple(dict.fromkeys(errors)), registry_sha256)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the evidence-bound Clean-v3 paper")
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_paper(args.paper, args.registry)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result.receipt(args.paper), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="ascii",
    )
    print(f"passed={str(result.passed).lower()} errors={len(result.errors)}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
