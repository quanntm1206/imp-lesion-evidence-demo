from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from lesion_robustness.release_manifest import (
    DEFAULT_MANIFEST,
    paper_projection,
)


EVIDENCE_CLASSES = {
    "protected_validation",
    "train_screen",
    "legacy_patient_contaminated",
}


@dataclass(frozen=True)
class EvidenceSources:
    loop191: Path
    loop192: Path
    loop206: Path
    loop170_locked_panel: Path
    loop170_bootstrap: Path


def canonical_json_bytes(payload: object) -> bytes:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return (encoded + "\n").encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence source must contain a JSON object: {path}")
    return payload


def _finite(value: object, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _relative_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence source is outside project root: {path}") from exc


def _source_rows(sources: EvidenceSources, project_root: Path) -> list[dict[str, str]]:
    entries = (
        ("loop191_report", sources.loop191),
        ("loop192_report", sources.loop192),
        ("loop206_report", sources.loop206),
        ("loop170_locked_panel", sources.loop170_locked_panel),
        ("loop170_bootstrap", sources.loop170_bootstrap),
    )
    rows = []
    for source_id, path in entries:
        if not path.is_file():
            raise FileNotFoundError(f"missing evidence source: {path}")
        rows.append(
            {
                "source_id": source_id,
                "path": _relative_path(path, project_root),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _metrics(payload: Mapping[str, Any], prefix: str) -> dict[str, float]:
    return {
        "robust_dice": _finite(payload["dice"], f"{prefix}.dice"),
        "robust_iou": _finite(payload["iou"], f"{prefix}.iou"),
        "robust_precision": _finite(payload["precision"], f"{prefix}.precision"),
        "robust_recall": _finite(payload["recall"], f"{prefix}.recall"),
        "robust_bf1": _finite(payload["boundary_f1"], f"{prefix}.boundary_f1"),
    }


def _loop191_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparison = paper_projection()["comparisons"][0]
    imp_model_id = str(comparison["left_model_id"])
    if payload.get("loop") != 191 or payload.get("test_opened") is not False:
        raise ValueError("Loop191 source is not the sealed validation report")
    integrity = payload.get("pre_pilot_integrity", {}).get("clean_v3", {})
    if integrity.get("status") != "complete" or integrity.get("identity_overlap_violations") != 0:
        raise ValueError("Loop191 source lacks a passing Clean-v3 integrity audit")
    controls = [
        row
        for row in payload.get("candidates", [])
        if row.get("id") == imp_model_id and row.get("role") == "control"
    ]
    if len(controls) != 1:
        raise ValueError("Loop191 control observation is missing or duplicated")
    robust = controls[0].get("metrics", {}).get("robust_mean", {})
    return {
        "model_id": imp_model_id,
        "display_name": "IMP-SegFormer-B3",
        "dataset_version": "Clean-v3",
        "partition": "validation",
        "evidence_class": "protected_validation",
        "scientific_comparable": True,
        "seed_count": 1,
        "metric_contract": "legacy_nearest_384_t2",
        "source_ids": ["loop191_report"],
        **_metrics(robust, "loop191.robust_mean"),
        "limitations": [
            "single-run validation point estimate",
            "protected test-v3 not opened",
            "older metric geometry contract",
        ],
    }


def _loop192_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparison = paper_projection()["comparisons"][0]
    nnunet_model_id = str(comparison["right_model_id"])
    if payload.get("loop") != 192 or payload.get("test_opened") is not False:
        raise ValueError("Loop192 source is not the sealed validation report")
    if payload.get("candidate_id") != nnunet_model_id:
        raise ValueError("Loop192 candidate identity changed")
    protocol = payload.get("evaluation_protocol", {})
    if protocol.get("model_image_size") != [256, 256] or protocol.get("metric_image_size") != [384, 384]:
        raise ValueError("Loop192 geometry contract changed")
    robust = payload.get("candidate", {}).get("robust_mean", {})
    return {
        "model_id": nnunet_model_id,
        "display_name": "nnU-Net v2",
        "dataset_version": "Clean-v3",
        "partition": "validation",
        "evidence_class": "protected_validation",
        "scientific_comparable": True,
        "seed_count": 1,
        "metric_contract": "legacy_nearest_384_t2",
        "source_ids": ["loop192_report"],
        **_metrics(robust, "loop192.robust_mean"),
        "limitations": [
            "single-run validation point estimate",
            "protected test-v3 not opened",
            "256x256 prediction resized to 384x384 metric canvas",
        ],
    }


def _loop206_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("loop") != 206 or payload.get("protected_panels_sealed") is not True:
        raise ValueError("Loop206 source does not prove sealed protected panels")
    if payload.get("evidence_validation", {}).get("passed") is not True:
        raise ValueError("Loop206 evidence validation did not pass")
    dice = payload.get("bootstrap", {}).get("dice", {})
    bf1 = payload.get("bootstrap", {}).get("boundary_f1", {})
    if dice.get("paired_seed_count") != 3 or bf1.get("paired_seed_count") != 3:
        raise ValueError("Loop206 evidence must contain three paired seeds")
    if dice.get("group_count") != 76 or bf1.get("group_count") != 76:
        raise ValueError("Loop206 evidence must contain 76 paired groups")
    return {
        "model_id": "L206-contour-vs-control",
        "display_name": "Loop206 contour channel minus zero-channel control",
        "dataset_version": "Clean-v3",
        "partition": "train_screen",
        "evidence_class": "train_screen",
        "scientific_comparable": False,
        "seed_count": 3,
        "metric_contract": "loop206_train_screen_three_corruptions_t2",
        "source_ids": ["loop206_report"],
        "robust_dice_delta": _finite(dice["point_delta"], "loop206.dice.point_delta"),
        "robust_dice_ci95": [
            _finite(dice["ci95_lower"], "loop206.dice.ci95_lower"),
            _finite(dice["ci95_upper"], "loop206.dice.ci95_upper"),
        ],
        "robust_bf1_delta": _finite(bf1["point_delta"], "loop206.bf1.point_delta"),
        "robust_bf1_ci95": [
            _finite(bf1["ci95_lower"], "loop206.bf1.ci95_lower"),
            _finite(bf1["ci95_upper"], "loop206.bf1.ci95_upper"),
        ],
        "group_count": 76,
        "corruption_count": 3,
        "bootstrap_resamples": int(dice["resamples"]),
        "limitations": [
            "train-screen evidence only",
            "Clean-v3 validation, test-v3, and PH2 remained sealed",
            "negative result applies only to the tested contour channel",
        ],
    }


_PANEL_ROW = re.compile(
    r"^TEST\s*&\s*([^&]+?)\s*&\s*([-+0-9.]+)\s*&\s*([-+0-9.]+)\s*&\s*([-+0-9.]+)\s*&\s*([-+0-9.]+)\s*\\\\",
    re.MULTILINE,
)
_BOOTSTRAP_ROW = re.compile(
    r"^Dice IMP--nnU-Net\s*&\s*([-+0-9.]+)\s*&\s*\[([-+0-9.]+),\s*([-+0-9.]+)\]",
    re.MULTILINE,
)


def _loop170_observations(panel_text: str) -> list[dict[str, Any]]:
    ids = {
        "IMP-SegFormer-B3": "Loop170-IMP",
        "Vanilla SegFormer-B3": "Loop170-Vanilla",
        "EGE-UNet": "Loop170-EGE-UNet",
        "nnU-Net v2": "Loop170-nnU-Net-v2",
    }
    matches = _PANEL_ROW.findall(panel_text)
    if {method.strip() for method, *_ in matches} != set(ids):
        raise ValueError("Loop170 locked TEST panel is incomplete or changed")
    rows = []
    for method, robust_dice, clean_dice, robust_iou, robust_bf1 in matches:
        name = method.strip()
        rows.append(
            {
                "model_id": ids[name],
                "display_name": name,
                "dataset_version": "Clean-v2",
                "partition": "test_v2",
                "evidence_class": "legacy_patient_contaminated",
                "scientific_comparable": False,
                "seed_count": 1,
                "metric_contract": "clean_v2_legacy_locked_panel_v1",
                "source_ids": ["loop170_locked_panel", "loop170_bootstrap"],
                "robust_dice": _finite(robust_dice, f"loop170.{name}.robust_dice"),
                "clean_dice": _finite(clean_dice, f"loop170.{name}.clean_dice"),
                "robust_iou": _finite(robust_iou, f"loop170.{name}.robust_iou"),
                "robust_bf1": _finite(robust_bf1, f"loop170.{name}.robust_bf1"),
                "limitations": [
                    "three patient IDs cross splits",
                    "13 rows participate in cross-split patient contamination",
                    "historical operational evidence only",
                ],
            }
        )
    return sorted(rows, key=lambda row: row["model_id"])


def _loop170_comparison(bootstrap_text: str) -> dict[str, Any]:
    match = _BOOTSTRAP_ROW.search(bootstrap_text)
    if match is None:
        raise ValueError("Loop170 IMP--nnU-Net bootstrap row is missing")
    mean, lower, upper = match.groups()
    return {
        "comparison_id": "Loop170-IMP-minus-nnU-Net-v2",
        "evidence_class": "legacy_patient_contaminated",
        "scientific_comparable": False,
        "metric": "robust_dice",
        "point_delta": _finite(mean, "loop170.bootstrap.mean"),
        "ci95": [
            _finite(lower, "loop170.bootstrap.lower"),
            _finite(upper, "loop170.bootstrap.upper"),
        ],
        "source_ids": ["loop170_bootstrap"],
    }


def _registry_hash(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("registry_sha256", None)
    # Release identity rotates independently of source-evidence content.
    unsigned.pop("release_manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def build_registry(
    sources: EvidenceSources,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    source_rows = _source_rows(sources, root)
    loop191 = _loop191_observation(_load_json(sources.loop191))
    loop192 = _loop192_observation(_load_json(sources.loop192))
    loop206 = _loop206_observation(_load_json(sources.loop206))
    legacy = _loop170_observations(sources.loop170_locked_panel.read_text(encoding="utf-8"))
    registry: dict[str, Any] = {
        "schema_version": "imp.evidence_registry.v1",
        "scientific_sota_status": "not_established",
        "claim_policy": {
            "protected_test_claim_allowed": False,
            "cross_protocol_ranking_allowed": False,
            "clinical_use": False,
        },
        "dataset": {
            "clean_v3_rows": 2869,
            "clean_v3_split_counts": {"train": 2008, "validation": 431, "test": 430},
            "clean_v2_cross_split_patient_ids": 3,
            "clean_v2_cross_split_rows": 13,
        },
        "sources": source_rows,
        "observations": [loop191, loop192, loop206, *legacy],
        "comparisons": [
            {
                "comparison_id": "L192-minus-L191",
                "evidence_class": "protected_validation",
                "scientific_comparable": True,
                "metric": "robust_dice",
                "point_delta": loop192["robust_dice"] - loop191["robust_dice"],
                "ci95": None,
                "source_ids": ["loop191_report", "loop192_report"],
                "limitations": ["single-run point estimates", "no paired confidence interval"],
            },
            {
                "comparison_id": "L206-contour-minus-control",
                "evidence_class": "train_screen",
                "scientific_comparable": False,
                "metric": "robust_dice",
                "point_delta": loop206["robust_dice_delta"],
                "ci95": loop206["robust_dice_ci95"],
                "source_ids": ["loop206_report"],
            },
            _loop170_comparison(sources.loop170_bootstrap.read_text(encoding="utf-8")),
        ],
        "release_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
    }
    registry["registry_sha256"] = _registry_hash(registry)
    validate_registry(registry)
    return registry


def validate_registry(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "imp.evidence_registry.v1":
        raise ValueError("unsupported evidence registry schema")
    if payload.get("scientific_sota_status") != "not_established":
        raise ValueError("scientific SOTA must remain not_established")
    release_digest = str(payload.get("release_manifest_sha256", ""))
    if len(release_digest) != 64 or not re.fullmatch(r"[0-9a-f]{64}", release_digest):
        raise ValueError("evidence registry release manifest digest mismatch")
    if release_digest != sha256_file(DEFAULT_MANIFEST):
        raise ValueError("evidence registry release manifest digest mismatch")
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("evidence registry requires five sources")
    source_ids = set()
    for source in sources:
        source_id = str(source.get("source_id", ""))
        digest = str(source.get("sha256", ""))
        source_path = Path(str(source.get("path", "")))
        if not source_id or source_id in source_ids:
            raise ValueError("evidence source IDs must be unique")
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("evidence source paths must be portable")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("evidence source SHA-256 is malformed")
        source_ids.add(source_id)
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("evidence registry has no observations")
    model_ids = set()
    for row in observations:
        model_id = str(row.get("model_id", ""))
        evidence_class = str(row.get("evidence_class", ""))
        if not model_id or model_id in model_ids:
            raise ValueError("model IDs must be non-empty and unique")
        if evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"unknown evidence class: {evidence_class}")
        if evidence_class == "legacy_patient_contaminated" and row.get("scientific_comparable") is not False:
            raise ValueError("legacy evidence cannot be scientifically comparable")
        if evidence_class == "train_screen" and row.get("partition") != "train_screen":
            raise ValueError("train-screen evidence cannot use a protected partition")
        if int(row.get("seed_count", 0)) < 1:
            raise ValueError("seed_count must be positive")
        for key, value in row.items():
            if key.startswith("robust_") and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite observation metric: {model_id}.{key}")
        if not set(row.get("source_ids", [])) <= source_ids:
            raise ValueError("observation references an unknown source")
        model_ids.add(model_id)
    expected_hash = str(payload.get("registry_sha256", ""))
    if len(expected_hash) != 64 or expected_hash != _registry_hash(payload):
        raise ValueError("evidence registry hash mismatch")


def write_registry(payload: Mapping[str, Any], output: str | Path) -> None:
    validate_registry(payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(destination)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the protocol-bound IMP evidence registry")
    parser.add_argument("--loop191", type=Path, required=True)
    parser.add_argument("--loop192", type=Path, required=True)
    parser.add_argument("--loop206", type=Path, required=True)
    parser.add_argument("--loop170-panel", type=Path, required=True)
    parser.add_argument("--loop170-bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    registry = build_registry(
        EvidenceSources(
            loop191=args.loop191,
            loop192=args.loop192,
            loop206=args.loop206,
            loop170_locked_panel=args.loop170_panel,
            loop170_bootstrap=args.loop170_bootstrap,
        ),
        project_root=args.project_root,
    )
    write_registry(registry, args.output)
    print(
        "registry_status=valid "
        f"scientific_sota_status={registry['scientific_sota_status']} "
        f"sources={len(registry['sources'])} observations={len(registry['observations'])} "
        f"registry_sha256={registry['registry_sha256']}"
    )
