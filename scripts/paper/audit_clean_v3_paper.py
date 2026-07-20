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
    r"\\(?:cite|citep|citet|textcite|parencite|autocite)\*?\s*"
    r"(?:\[[^]]*\]\s*)*\{([^}]+)\}",
    re.IGNORECASE,
)
_BIB_KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z_0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_METRIC = re.compile(
    r"robust[- ]?dice|clean[- ]?dice|boundary[- ]?f1|robust[- ]?iou|"
    r"(?:^|\W)(?:dice|iou|precision|recall|hd95|assd)(?:\W|$)",
    re.IGNORECASE,
)
_RESULT_SIGNAL = re.compile(
    r"point estimate|confidence interval|candidate-minus-control|\bdelta\b|"
    r"obtained|was|were|is|are|higher|lower|reduces?|increases?|decreases?|changes?|"
    r"improvement|difference|versus|\bci\b",
    re.IGNORECASE,
)
_PROTOCOL = re.compile(r"seed|group|resample|corruption", re.IGNORECASE)
_INCLUDE_GRAPHICS = re.compile(
    r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}"
)
_TABLE_INPUT = re.compile(r"\\input\s*\{(tables/[^}]+)\}")
_UNFINISHED = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|\?\?")
_CLAIM_TERMS = re.compile(
    r"state[ -]of[ -]the[ -]art|\bsota\b|statistical(?:ly)?[ -]superior(?:ity)?|"
    r"clinical[- ]grade|clinical validation|clinical system|clinical use|diagnostic(?: claim| system| use)?|"
    r"protected[- ]test (?:accuracy|dice|iou|bf1|metric|score|performance|result|evidence|claim)|"
    r"significantly outperform(?:s|ed|ing)?|significant improvement|significant superiority",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|unavailable|sealed|prevent(?:s|ed)?|"
    r"does not|do not|did not|cannot|is not|are not|has not|have not|rather than|"
    r"from being ranked)\b",
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


def _number_is_supported(token: str, supported: Iterable[float]) -> bool:
    value = float(token.replace(",", ""))
    if value in {95.0}:  # Confidence-level notation, not an empirical result.
        return True
    for candidate in supported:
        for transformed in (candidate, abs(candidate), round(candidate, 4), round(abs(candidate), 4)):
            if value == transformed:
                return True
    return False


def _sentences(text: str) -> Iterable[str]:
    return re.split(r"(?<=[.!?])\s+|\n+", text)


def _clauses(text: str) -> Iterable[str]:
    for sentence in _sentences(text):
        yield from re.split(
            r"\s*(?:;|--|—|\bbut\b|\bhowever\b|\byet\b|\bwhereas\b)\s*",
            sentence,
            flags=re.IGNORECASE,
        )


def _claim_clauses(text: str) -> Iterable[str]:
    for sentence in _sentences(text):
        if re.match(r"\s*no\s+\w+\s+is\s+presented\s+as\b", sentence, re.IGNORECASE):
            yield sentence
            continue
        yield from re.split(
            r"\s*(?:[,;:]|--|—|\bbut\b|\bhowever\b|\byet\b|\bwhereas\b|\balthough\b|\bwhile\b)\s*",
            sentence,
            flags=re.IGNORECASE,
        )


def _claim_is_negated(clause: str) -> bool:
    return bool(_NEGATION.search(clause))


def _check_claims(path: Path, text: str, root: Path, errors: list[str]) -> None:
    for clause in _claim_clauses(text):
        if _CLAIM_TERMS.search(clause) and not _claim_is_negated(clause):
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
    registry_path = manifest.get("evidence_registry_path")
    registry_hash = manifest.get("evidence_registry_sha256")
    if not isinstance(registry_path, str) or not isinstance(registry_hash, str):
        errors.append("missing evidence mapping")
    elif _resolve_contained(root, registry_path) is None:
        errors.append("unsafe manifest path")
    for category, label in (("figures", "figure"), ("tables", "table")):
        entries = manifest.get(category, {})
        if not isinstance(entries, dict):
            errors.append(f"invalid {category} manifest")
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                errors.append(f"invalid {label} manifest entry")
                continue
            _check_declared_hashes(paper, entry, label, errors)
    return manifest


def _resolve_contained(base: Path, value: str) -> Path | None:
    if (
        Path(value).is_absolute()
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or value.startswith("\\\\")
        or ".." in re.split(r"[\\/]", value)
    ):
        return None
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def _check_declared_hashes(
    paper: Path, entry: dict[str, Any], primary_label: str, errors: list[str]
) -> None:
    pairs = [("path", "sha256", primary_label)]
    pairs.extend(
        (key, f"{key[:-5]}_sha256", "source")
        for key in entry
        if key.endswith("_path")
    )
    semantic_hashes = {
        "evidence_registry_sha256",
        "external_runtime_bundle_sha256",
        "provenance_manifest_sha256",
        "provenance_receipt_sha256",
    }
    pairs.extend(
        (f"{key[:-7]}_path", key, "source")
        for key in entry
        if key.endswith("_sha256") and key not in semantic_hashes and key != "sha256"
    )
    for path_key, hash_key, label in dict.fromkeys(pairs):
        _check_hashed_artifact(paper, entry, path_key, hash_key, label, errors)


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
    if not isinstance(source, str) or not isinstance(expected, str):
        errors.append(f"missing {label} hash")
        return
    artifact = _resolve_contained(paper, source)
    if artifact is None:
        errors.append("unsafe manifest path")
    elif not artifact.is_file():
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
        candidate = _resolve_contained(root, relative)
        if candidate is None or not candidate.is_file() or _sha256(candidate) != expected:
            errors.append(f"source hash drift: {source.get('source_id', 'unknown')}")


def _check_loop170_labels(tex_files: Iterable[Path], root: Path, errors: list[str]) -> None:
    for path in tex_files:
        text = _read(path, errors)
        if re.search(r"loop170", text, re.IGNORECASE) and _NUMBER.search(text):
            label = "legacy_patient_contaminated"
            if label not in text.replace("\\_", "_"):
                errors.append(f"unlabeled Loop170 values: {_relative(path, root)}")


def _numbers_for_field(rows: object, field: str) -> set[float]:
    values: set[float] = set()
    if not isinstance(rows, list):
        return values
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.add(float(value))
        elif isinstance(value, list):
            values.update(
                float(item)
                for item in value
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            )
    return values


def _metric_fields(metric: str, evidence: str) -> tuple[str, ...]:
    if evidence == "point":
        return {
            "dice": ("robust_dice", "clean_dice"),
            "iou": ("robust_iou",),
            "bf1": ("robust_bf1",),
            "precision": ("robust_precision",),
            "recall": ("robust_recall",),
            "hd95": (),
            "assd": (),
        }[metric]
    if evidence == "delta":
        return {
            "dice": ("robust_dice_delta",),
            "iou": (),
            "bf1": ("robust_bf1_delta",),
            "precision": (),
            "recall": (),
            "hd95": (),
            "assd": (),
        }[metric]
    return {
        "dice": ("robust_dice_ci95",),
        "iou": (),
        "bf1": ("robust_bf1_ci95",),
        "precision": (),
        "recall": (),
        "hd95": (),
        "assd": (),
    }[metric]


def _metric_name(clause: str, position: int) -> str | None:
    aliases = (
        ("bf1", r"boundary[- ]?f1|\bbf1\b"),
        ("dice", r"(?:robust|clean)[- ]?dice|\bdice\b"),
        ("iou", r"(?:robust[- ]?)?iou\b"),
        ("precision", r"\bprecision\b"),
        ("recall", r"\brecall\b"),
        ("hd95", r"\bhd95\b"),
        ("assd", r"\bassd\b"),
    )
    matches = [
        (match.start(), name)
        for name, pattern in aliases
        for match in re.finditer(pattern, clause, re.IGNORECASE)
    ]
    before = [item for item in matches if item[0] <= position]
    if before:
        return max(before)[1]
    return min(matches)[1] if matches else None


def _claim_evidence(path: Path, clause: str, position: int) -> set[str]:
    if path.name == "loop206_ablation.tex":
        return {"delta", "ci"}
    if clause.rfind("[", 0, position) > clause.rfind("]", 0, position):
        return {"ci"}
    prior = clause[:position].lower()
    markers = [
        (match.start(), "ci")
        for match in re.finditer(r"confidence interval|\bci\b", prior)
    ]
    markers.extend(
        (match.start(), "delta")
        for match in re.finditer(
            r"\b(?:delta|difference|change)\b|candidate-minus-control|"
            r"reduc(?:e|es|ed|ing)?\b|increas(?:e|es|ed|ing)?\b|"
            r"decreas(?:e|es|ed|ing)?\b|chang(?:e|es|ed|ing)?\b|"
            r"\b(?:higher|lower)\b[^.]{0,32}\bby\b",
            prior,
        )
    )
    markers.extend(
        (match.start(), "point")
        for match in re.finditer(r"point estimate|\bobtained\b", prior)
    )
    return {max(markers)[1]} if markers else {"point"}


def _metric_values(
    registry: dict[str, Any], metric: str | None, evidence: set[str]
) -> set[float]:
    observations = registry.get("observations")
    comparisons = registry.get("comparisons")
    values: set[float] = set()
    metrics = (metric,) if metric is not None else ("dice", "bf1")
    for current_metric in metrics:
        for kind in evidence:
            for field in _metric_fields(current_metric, kind):
                values.update(_numbers_for_field(observations, field))
    if metric == "dice" and isinstance(comparisons, list):
        for comparison in comparisons:
            if not isinstance(comparison, dict) or comparison.get("metric") != "robust_dice":
                continue
            if "delta" in evidence and isinstance(comparison.get("point_delta"), (int, float)):
                values.add(float(comparison["point_delta"]))
            if "ci" in evidence and isinstance(comparison.get("ci95"), list):
                values.update(
                    float(item)
                    for item in comparison["ci95"]
                    if isinstance(item, (int, float))
                )
    values.update(abs(value) for value in tuple(values))
    return values


def _protocol_values(
    registry: dict[str, Any], clause: str, start: int, end: int
) -> set[float]:
    after = clause[end : end + 48].lower()
    observations = registry.get("observations")
    fields = []
    if re.match(r"\s*(?:-| )\s*(?:paired\s+)?seeds?\b", after):
        fields.append("seed_count")
    if re.match(r"\s*(?:-| )\s*groups?\b", after):
        fields.append("group_count")
    if re.match(r"\s+(?:[a-z-]+\s+){0,3}resamples?\b", after):
        fields.append("bootstrap_resamples")
    if re.match(r"\s+(?:[a-z-]+\s+){0,2}corruptions?\b", after):
        fields.append("corruption_count")
    values: set[float] = set()
    for field in fields:
        values.update(_numbers_for_field(observations, field))
    return values


def _is_numeric_claim(path: Path, clause: str) -> bool:
    if path.suffix != ".tex":
        return False
    if "tables" in {part.lower() for part in path.parts}:
        return bool(_METRIC.search(clause) or _PROTOCOL.search(clause))
    return bool(_METRIC.search(clause) and _RESULT_SIGNAL.search(clause))


def _check_result_numbers(
    tex_files: Iterable[Path], registry: dict[str, Any], root: Path, errors: list[str]
) -> None:
    for path in tex_files:
        for clause in _clauses(_read(path, errors)):
            if not _is_numeric_claim(path, clause):
                continue
            for match in _NUMBER.finditer(clause):
                protocol = _protocol_values(registry, clause, match.start(), match.end())
                metric = _metric_values(
                    registry,
                    _metric_name(clause, match.start()),
                    _claim_evidence(path, clause, match.start()),
                )
                supported = protocol or metric
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


def _normalized_tex_path(value: str, suffix: str) -> str:
    path = Path(value.replace("\\", "/"))
    return path.as_posix() if path.suffix else f"{path.as_posix()}{suffix}"


def _check_manifest_references(
    manifest: dict[str, Any], tex_files: Iterable[Path], errors: list[str]
) -> None:
    figures = manifest.get("figures", {})
    tables = manifest.get("tables", {})
    figure_paths = {
        str(entry.get("path"))
        for entry in figures.values()
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    } if isinstance(figures, dict) else set()
    table_paths = {
        str(entry.get("path"))
        for entry in tables.values()
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    } if isinstance(tables, dict) else set()
    for path in tex_files:
        text = _read(path, errors)
        for value in _INCLUDE_GRAPHICS.findall(text):
            if _normalized_tex_path(value.strip(), ".pdf") not in figure_paths:
                errors.append("unmapped figure input")
        for value in _TABLE_INPUT.findall(text):
            if _normalized_tex_path(value.strip(), ".tex") not in table_paths:
                errors.append("unmapped table input")


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
        manifest = _check_manifest(paper_path, registry_payload, root, errors)
        if manifest is not None:
            _check_manifest_references(manifest, tex_files, errors)
        _check_source_hashes(registry_payload, root, errors)
        _check_loop170_labels(tex_files, root, errors)
        _check_result_numbers(tex_files, registry_payload, root, errors)
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
