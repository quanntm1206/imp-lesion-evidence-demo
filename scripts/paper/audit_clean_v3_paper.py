"""Fail closed when the evidence-bound paper or demo drifts from its registry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import re
import subprocess
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
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAIM_TERMS = re.compile(
    r"state[ -]of[ -]the[ -]art|\bsota\b|statistical(?:ly)?[ -]superior(?:ity)?|"
    r"clinical[- ]grade|clinical validation|clinical system|clinical use|diagnostic(?: claim| system| use)?|"
    r"protected[- ]test (?:accuracy|dice|iou|bf1|boundary[- ]f1|precision|recall|hd95|assd|"
    r"metric|score|performance|result|evidence|claim)|"
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
    source_verification: str = "strict"
    missing_source_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

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
            "source_verification": self.source_verification,
            "missing_source_ids": list(self.missing_source_ids),
            "warnings": list(self.warnings),
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
        subordinate = re.match(
            r"\s*(?:although|while|though|even\s+though)\b(.+?),\s*(.+)",
            sentence,
            re.IGNORECASE,
        )
        if subordinate:
            yield subordinate.group(1)
            yield subordinate.group(2)
            continue
        yield from re.split(
            r"\s*(?:;|:|--|—|\bbut\b|\bhowever\b|\byet\b|\bwhereas\b)\s*",
            sentence,
            flags=re.IGNORECASE,
        )


_COORDINATION_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9_][A-Za-z0-9_'-]*")
_BARE_COMPLEMENT_ARTICLES = frozenset({"a", "an", "the"})


def _is_bare_claim_continuation(value: str) -> bool:
    tokens = _COORDINATION_TOKEN.findall(value)
    return all(token.lower() in _BARE_COMPLEMENT_ARTICLES for token in tokens)


def _claim_is_negated(clause: str, start: int, end: int) -> bool:
    prefix = clause[:start]
    boundaries: list[re.Match[str]] = []
    for coordination in re.finditer(r"\b(?:and|or)\b", prefix, re.IGNORECASE):
        right = prefix[coordination.end() :]
        if not _is_bare_claim_continuation(right):
            boundaries.append(coordination)
    scoped_prefix = prefix[boundaries[-1].start() :] if boundaries else prefix
    if _NEGATION.search(scoped_prefix):
        return True
    suffix = clause[end:]
    return bool(
        re.match(
            r"\s+(?:remains?|is|are|was|were)\s+(?:sealed|unavailable)\b|"
            r"\s+(?:is|are|was|were)\s+not\b",
            suffix,
            re.IGNORECASE,
        )
    )


def _check_claims(path: Path, text: str, root: Path, errors: list[str]) -> None:
    for clause in _claim_clauses(text):
        for match in _CLAIM_TERMS.finditer(clause):
            if not _claim_is_negated(clause, match.start(), match.end()):
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
    _check_paper_pdf(paper, manifest, errors)
    return manifest


def _pdf_page_count(path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("pdfinfo failed")
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    if match is None or int(match.group(1)) < 1:
        raise ValueError("pdfinfo returned no valid page count")
    return int(match.group(1))


def _check_paper_pdf(
    paper: Path, manifest: dict[str, Any], errors: list[str]
) -> None:
    entry = manifest.get("paper_pdf")
    if not isinstance(entry, dict):
        errors.append("missing paper PDF binding")
        return
    relative = entry.get("path")
    expected_hash = entry.get("sha256")
    expected_pages = entry.get("pages")
    if (
        relative != "main.pdf"
        or not isinstance(expected_hash, str)
        or type(expected_pages) is not int
        or expected_pages < 1
    ):
        errors.append("missing paper PDF binding")
        return
    pdf = _resolve_contained(paper, relative)
    if pdf is None or pdf != (paper / "main.pdf").resolve() or not pdf.is_file():
        errors.append("missing paper PDF binding")
        return
    if _sha256(pdf) != expected_hash:
        errors.append("paper PDF hash drift")
    try:
        if _pdf_page_count(pdf) != expected_pages:
            errors.append("paper PDF page drift")
    except (OSError, ValueError, subprocess.SubprocessError):
        errors.append("paper PDF inspection failed")


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
    registry: dict[str, Any],
    root: Path,
    errors: list[str],
    source_verification: str,
) -> tuple[str, ...]:
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("missing evidence mapping")
        return ()
    missing_source_ids: list[str] = []
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
        source_id = str(source.get("source_id", "unknown"))
        if candidate is None:
            errors.append(f"source hash drift: {source_id}")
        elif not candidate.is_file():
            missing_source_ids.append(source_id)
            if source_verification == "strict":
                errors.append(f"source hash drift: {source_id}")
        elif _sha256(candidate) != expected:
            errors.append(f"source hash drift: {source_id}")
    return tuple(sorted(missing_source_ids))


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


def _record_loop_ids(record: dict[str, Any]) -> set[str]:
    values = [
        str(record.get("model_id", "")),
        str(record.get("comparison_id", "")),
        *(str(value) for value in record.get("source_ids", []) if isinstance(value, str)),
    ]
    return {
        match
        for value in values
        for match in re.findall(r"\b(?:loop|l)[- _]?(\d{3})\b", value.lower())
    }


def _normalized_phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).replace("\\_", "_").lower()))


def _identity_patterns(record: dict[str, Any]) -> set[str]:
    patterns = {
        rf"\b(?:loop|l)[- _]?{identifier}\b"
        for identifier in _record_loop_ids(record)
    }
    for field in ("display_name", "model_id", "comparison_id"):
        words = re.findall(r"[a-z0-9]+", str(record.get(field, "")).lower())
        if words:
            patterns.add(r"\b" + r"[- _\\]*".join(map(re.escape, words)) + r"\b")
    display = _normalized_phrase(record.get("display_name", ""))
    if display.startswith("imp segformer"):
        patterns.add(r"\\impmodel\b")
        patterns.add(r"\b(?:preprocessing[- ]aware\s+)?mit[- ]?b3\s+u[- ]?net\s+control\b")
    return patterns


_MODEL_IDENTIFIER = re.compile(
    r"\b(?:loop|l)[- _]?\d+\b|"
    r"\b(?=[A-Za-z0-9-]*[A-Z])(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9-]*\b|"
    r"\b[A-Z]?[a-z]+[A-Z][A-Za-z0-9-]*\b|"
    r"\b[A-Z]{2,}(?:-[A-Za-z0-9]+)+\b|"
    r"\b[A-Z][A-Za-z0-9-]+\s+(?:model|system)\b"
)
_LOWERCASE_ARCHITECTURE_IDENTIFIER = re.compile(
    r"\b(?:resnet\d*|nn[- ]?u[- ]?net|u[- ]?net|segformer(?:-?[a-z0-9]+)*|"
    r"medsam|sam|vit|cnn)\b",
    re.IGNORECASE,
)


def _claim_names_model(clause: str, records: list[dict[str, Any]]) -> bool:
    without_metrics = _METRIC.sub(" ", clause)
    if (
        _MODEL_IDENTIFIER.search(without_metrics) is not None
        or _LOWERCASE_ARCHITECTURE_IDENTIFIER.search(without_metrics) is not None
    ):
        return True
    normalized = _normalized_phrase(clause)
    markers = {
        _normalized_phrase(record.get(field, ""))
        for record in records
        for field in ("evidence_class", "partition", "metric_contract")
    }
    positions = [normalized.find(marker) for marker in markers if marker and marker in normalized]
    if not positions:
        return False
    prefix = normalized[: min(positions)].split()
    while prefix and prefix[0] in {
        "a",
        "an",
        "at",
        "for",
        "from",
        "in",
        "on",
        "the",
        "under",
    }:
        prefix.pop(0)
    if prefix[:2] == ["metric", "contract"]:
        prefix = prefix[2:]
    return bool(prefix)


def _identity_scoped_records(
    records: list[dict[str, Any]], clause: str, position: int
) -> list[dict[str, Any]]:
    before: list[tuple[int, int]] = []
    connected_after: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        for pattern in _identity_patterns(record):
            for match in re.finditer(pattern, clause, re.IGNORECASE):
                if match.end() <= position:
                    before.append((position - match.end(), index))
                elif match.start() >= position:
                    connector = clause[position : match.start()]
                    if re.fullmatch(
                        r"[-+0-9.eE]+\s+(?:for|by|from)\s+",
                        connector,
                        re.IGNORECASE,
                    ):
                        connected_after.append((match.start() - position, index))
                else:
                    before.append((0, index))
    distances = connected_after or before
    if not distances:
        return [] if _claim_names_model(clause, records) else records
    nearest = min(distance for distance, _ in distances)
    selected = {index for distance, index in distances if distance == nearest}
    return [record for index, record in enumerate(records) if index in selected]


def _record_matches_claim(record: dict[str, Any], clause: str) -> bool:
    normalized = _normalized_phrase(clause)
    for field in ("evidence_class", "partition", "metric_contract"):
        value = _normalized_phrase(record.get(field, ""))
        if not value:
            continue
        known = record.get(f"_known_{field}", set())
        mentioned = {item for item in known if item and item in normalized}
        if mentioned and value not in mentioned:
            return False
    return True


def _point_fields(metric: str, clause: str) -> tuple[str, ...]:
    normalized = clause.replace("\\_", "_").lower()
    if metric == "dice":
        if re.search(r"\bclean[- ]?dice\b", normalized):
            return ("clean_dice",)
        if re.search(r"\brobust[- ]?dice\b", normalized):
            return ("robust_dice",)
    return _metric_fields(metric, "point")


def _metric_values(
    registry: dict[str, Any],
    metric: str | None,
    evidence: set[str],
    clause: str,
    position: int,
) -> set[float]:
    observations = registry.get("observations")
    comparisons = registry.get("comparisons")
    values: set[float] = set()
    metrics = (metric,) if metric is not None else ("dice", "bf1")
    records = [value for value in observations or [] if isinstance(value, dict)]
    known_fields = {
        field: {_normalized_phrase(record.get(field, "")) for record in records}
        for field in ("evidence_class", "partition", "metric_contract")
    }
    scoped_records: list[dict[str, Any]] = []
    for record in _identity_scoped_records(records, clause, position):
        scoped = dict(record)
        scoped["_all_records"] = records
        for field, known_values in known_fields.items():
            scoped[f"_known_{field}"] = known_values
        if _record_matches_claim(scoped, clause):
            scoped_records.append(record)
    for current_metric in metrics:
        for kind in evidence:
            fields = (
                _point_fields(current_metric, clause)
                if kind == "point"
                else _metric_fields(current_metric, kind)
            )
            for field in fields:
                values.update(_numbers_for_field(scoped_records, field))
    if metric == "dice" and isinstance(comparisons, list):
        comparison_records = [
            value for value in comparisons if isinstance(value, dict)
        ]
        for comparison in _identity_scoped_records(
            comparison_records, clause, position
        ):
            if (
                not isinstance(comparison, dict)
                or comparison.get("metric") != "robust_dice"
                or not _record_matches_claim(
                    {
                        **comparison,
                        "_known_evidence_class": {
                            _normalized_phrase(item.get("evidence_class", ""))
                            for item in comparisons
                            if isinstance(item, dict)
                        },
                        "_known_partition": set(),
                        "_known_metric_contract": set(),
                        "_all_records": [],
                    },
                    clause,
                )
            ):
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
                    clause,
                    match.start(),
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
    bundle_authorization = payload.get("display_authorization")
    required_hash_binding = (
        "sha256_raw+sha256_rgb+mask_sha256_raw+mask_sha256_binary"
    )
    if (
        not isinstance(bundle_authorization, dict)
        or bundle_authorization.get("hash_binding") != required_hash_binding
        or not _SHA256.fullmatch(
            str(bundle_authorization.get("mask_bindings_sha256", ""))
        )
    ):
        errors.append("unbound demo mask authorization")
    authorized_count = (
        bundle_authorization.get("authorized_sample_count")
        if isinstance(bundle_authorization, dict)
        else None
    )
    if authorized_count != 3 or len(entries) != authorized_count:
        errors.append("demo authorization count mismatch")
    mask_bindings: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("invalid demo receipt bundle")
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict) or any(
            not isinstance(metrics.get(arm), dict) for arm in ("control", "candidate")
        ):
            errors.append("missing demo metrics")
        authorization = entry.get("display_authorization")
        if (
            not isinstance(authorization, dict)
            or authorization.get("mask_variant") != "challenge_ground_truth"
            or authorization != bundle_authorization
        ):
            errors.append("hidden no-GT metrics")
        binding = entry.get("ground_truth_binding")
        required_mask_fields = (
            "mask_sha256_raw",
            "mask_sha256_binary",
            "mask_sha256_runtime",
        )
        if not isinstance(binding, dict) or any(
            not _SHA256.fullmatch(str(binding.get(field, "")))
            for field in required_mask_fields
        ):
            errors.append("unbound demo ground truth")
            continue
        sample = entry.get("sample")
        if not isinstance(sample, dict):
            errors.append("unbound demo ground truth")
            continue
        mask_bindings.append(
            {
                "group_key": str(sample.get("group_key", "")),
                "sample_id": str(sample.get("sample_id", "")),
                "mask_sha256_raw": str(binding["mask_sha256_raw"]),
                "mask_sha256_binary": str(binding["mask_sha256_binary"]),
            }
        )
    if len(mask_bindings) != len(entries):
        errors.append("unbound demo mask authorization")
    elif isinstance(bundle_authorization, dict):
        encoded = json.dumps(
            sorted(
                mask_bindings,
                key=lambda row: (row["sample_id"], row["group_key"]),
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if bundle_authorization.get("mask_bindings_sha256") != hashlib.sha256(
            encoded
        ).hexdigest():
            errors.append("unbound demo mask authorization")


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


def audit_paper(
    paper: str | Path,
    registry: str | Path,
    *,
    source_verification: str = "strict",
) -> AuditResult:
    """Audit paper inputs without emitting absolute filesystem paths."""
    if source_verification not in {"strict", "registry-only"}:
        raise ValueError("source_verification must be strict or registry-only")
    paper_path = Path(paper).resolve()
    registry_path = Path(registry).resolve()
    root = paper_path.parents[1] if len(paper_path.parents) > 1 else paper_path.parent
    errors: list[str] = []
    registry_payload: dict[str, Any] | None = None
    missing_source_ids: tuple[str, ...] = ()
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
        missing_source_ids = _check_source_hashes(
            registry_payload, root, errors, source_verification
        )
        _check_loop170_labels(tex_files, root, errors)
        _check_result_numbers(tex_files, registry_payload, root, errors)
        _check_demo_receipts(paper_path, registry_payload, errors)
        registry_sha256 = str(registry_payload.get("registry_sha256"))
    else:
        registry_sha256 = None
    warnings = (
        ("source bytes unavailable; strict local release audit required",)
        if missing_source_ids and source_verification == "registry-only"
        else ()
    )
    return AuditResult(
        tuple(dict.fromkeys(errors)),
        registry_sha256,
        source_verification,
        missing_source_ids,
        warnings,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the evidence-bound Clean-v3 paper")
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--source-verification",
        choices=("strict", "registry-only"),
        default="strict",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_paper(
        args.paper,
        args.registry,
        source_verification=args.source_verification,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result.receipt(args.paper), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="ascii",
    )
    print(
        f"passed={str(result.passed).lower()} errors={len(result.errors)} "
        f"source_verification={result.source_verification} "
        f"missing_sources={len(result.missing_source_ids)}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
