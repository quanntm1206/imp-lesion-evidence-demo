from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from lesion_robustness.release_manifest import paper_projection

from lesion_robustness.evidence_registry import validate_registry


def _escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _paper_input_paths(paper_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in paper_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(paper_dir)
        if path.suffix.lower() in {".tex", ".bib", ".cls", ".sty"}:
            paths.append(path)
        elif path.suffix.lower() == ".pdf" and relative.parts[0] == "figures":
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(paper_dir).as_posix())


def _paper_input_sha256(
    paper_dir: str | Path, overrides: Mapping[str, bytes] | None = None
) -> str:
    root = Path(paper_dir)
    override_map = dict(overrides or {})
    paths = {path.relative_to(root).as_posix(): path for path in _paper_input_paths(root)}
    paths.update({relative: root / relative for relative in override_map})
    payload = {
        "release_manifest_sha256": paper_projection()["release_manifest_sha256"],
        "inputs": [
            {
                "path": relative,
                "sha256": _sha256_bytes(override_map[relative])
                if relative in override_map
                else _sha256(paths[relative]),
            }
            for relative in sorted(paths)
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256((encoded + "\n").encode("ascii")).hexdigest()


def paper_input_sha256(paper_dir: str | Path) -> str:
    return _paper_input_sha256(paper_dir)


def _trusted_pdfinfo_executable() -> Path:
    override = os.environ.get("IMP_PDFINFO_EXE")
    candidate = (
        Path(override)
        if override is not None
        else Path(sys.executable).resolve().parent.parent
        / "native"
        / "poppler"
        / "Library"
        / "bin"
        / "pdfinfo.exe"
    )
    try:
        resolved = candidate.resolve(strict=True)
        stat = resolved.lstat()
    except OSError as exc:
        raise ValueError("trusted pdfinfo executable unavailable") from exc
    if not resolved.is_absolute() or not resolved.is_file() or resolved.is_symlink():
        raise ValueError("trusted pdfinfo executable unavailable")
    if getattr(stat, "st_file_attributes", 0) & 0x400:
        raise ValueError("trusted pdfinfo executable unavailable")
    return resolved


def _inspect_pdf(path: Path) -> int:
    executable = _trusted_pdfinfo_executable()
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        capture_output=True,
        text=True,
        check=False,
        executable=str(executable),
    )
    if completed.returncode != 0:
        raise ValueError("paper PDF inspection failed")
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    if match is None or int(match.group(1)) < 1:
        raise ValueError("paper PDF inspection failed")
    return int(match.group(1))


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )


def promote_paper_pdf(
    paper_dir: str | Path,
    *,
    expected_paper_input_sha256: str,
    visual_review_passed: bool,
) -> Path:
    root = Path(paper_dir)
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if not isinstance(manifest, dict):
        raise ValueError("existing artifact manifest must be an object")
    if visual_review_passed is not True:
        raise ValueError("paper PDF visual review is required")
    current_digest = paper_input_sha256(root)
    if expected_paper_input_sha256 != current_digest:
        raise ValueError("paper inputs changed before PDF promotion")
    pdf = root / "main.pdf"
    if not pdf.is_file():
        raise ValueError("paper PDF is unavailable")
    latest_input_mtime = max(
        (path.stat().st_mtime_ns for path in _paper_input_paths(root)), default=0
    )
    if pdf.stat().st_mtime_ns < latest_input_mtime:
        raise ValueError("paper PDF is older than paper inputs")
    pages = _inspect_pdf(pdf)
    release_digest = paper_projection()["release_manifest_sha256"]
    manifest["paper_input_sha256"] = current_digest
    manifest["paper_pdf"] = {
        "path": "main.pdf",
        "sha256": _sha256(pdf),
        "pages": pages,
        "bytes": pdf.stat().st_size,
        "status": "current",
        "built_release_manifest_sha256": release_digest,
        "built_paper_input_sha256": current_digest,
        "inspection": {
            "pdfinfo": "passed",
            "visual_review": "passed",
        },
    }
    _write_manifest(manifest_path, manifest)
    return manifest_path


def _observation(registry: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    rows = [row for row in registry["observations"] if row["model_id"] == model_id]
    if len(rows) != 1:
        raise ValueError(f"expected one evidence observation for {model_id}")
    return rows[0]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _clean_v3_table(registry: Mapping[str, Any]) -> str:
    comparison = paper_projection()["comparisons"][0]
    rows = [
        _observation(registry, str(comparison["left_model_id"])),
        _observation(registry, str(comparison["right_model_id"])),
    ]
    body = "\n".join(
        f"{_escape(row['display_name'])} & {row['robust_dice']:.4f} & "
        f"{row['robust_iou']:.4f} & {row['robust_precision']:.4f} & "
        f"{row['robust_recall']:.4f} & {row['robust_bf1']:.4f} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Clean-v3 adaptive development-validation point estimates under the older geometry contract. Both rows are single-run, selection-optimistic evidence; no confidence interval or protected-test result is available.}}
\label{{tab:clean-v3-validation}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Model & Robust Dice & Robust IoU & Precision & Recall & BF1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _loop206_table(registry: Mapping[str, Any]) -> str:
    row = _observation(registry, "L206-contour-vs-control")
    dice_ci = row["robust_dice_ci95"]
    bf1_ci = row["robust_bf1_ci95"]
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Loop206 contour-channel minus zero-channel control on the train-screen protocol. After averaging three selected seeds and three views, 10,000 bootstrap resamples draw 76 groups as whole split-group clusters; the interval is conditional on those seeds.}}
\label{{tab:loop206-ablation}}
\begin{{tabular}}{{lrr}}
\toprule
Metric & Point delta & 95\% CI \\
\midrule
Robust Dice & {row['robust_dice_delta']:.4f} & [{dice_ci[0]:.4f}, {dice_ci[1]:.4f}] \\
Boundary F1 & {row['robust_bf1_delta']:.4f} & [{bf1_ci[0]:.4f}, {bf1_ci[1]:.4f}] \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _verified_source_receipt(
    registry: Mapping[str, Any], registry_file: Path, source_id: str
) -> tuple[Mapping[str, Any] | None, str]:
    matches = [source for source in registry["sources"] if source["source_id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected one evidence source for {source_id}")
    source = matches[0]
    reference = f"{source_id}@{source['sha256'][:12]}"
    project_root = registry_file.resolve().parents[2]
    source_path = project_root / str(source["path"])
    if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
        return None, reference
    try:
        receipt = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, reference
    if not isinstance(receipt, dict):
        return None, reference
    if (
        receipt.get("schema_version") != "loop206.final_closure.v1"
        or receipt.get("loop") != 206
        or receipt.get("evidence_validation", {}).get("passed") is not True
    ):
        return None, reference
    return receipt, reference


def _receipt_number(receipt: Mapping[str, Any] | None, *keys: str) -> float | None:
    value: object = receipt
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _format_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _format_interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "unavailable"
    return f"[{lower:.4f}, {upper:.4f}]"


def _loop206_gate_audit_table(
    registry: Mapping[str, Any], registry_file: Path
) -> str:
    receipt, source_reference = _verified_source_receipt(
        registry, registry_file, "loop206_report"
    )
    dice = _receipt_number(receipt, "bootstrap", "dice", "point_delta")
    dice_interval = _format_interval(
        _receipt_number(receipt, "bootstrap", "dice", "ci95_lower"),
        _receipt_number(receipt, "bootstrap", "dice", "ci95_upper"),
    )
    boundary = _receipt_number(receipt, "bootstrap", "boundary_f1", "point_delta")
    boundary_interval = _format_interval(
        _receipt_number(receipt, "bootstrap", "boundary_f1", "ci95_lower"),
        _receipt_number(receipt, "bootstrap", "boundary_f1", "ci95_upper"),
    )
    distance_values = (
        _receipt_number(receipt, "robust_deltas", "hd95"),
        _receipt_number(receipt, "robust_deltas", "assd"),
    )
    distance_observed = (
        "unavailable"
        if None in distance_values
        else f"{_format_number(distance_values[0])} / {_format_number(distance_values[1])}"
    )
    rows = (
        ("primary_improvement", "robust Dice delta", _format_number(dice), dice_interval),
        ("dice_noninferiority", "robust Dice delta", _format_number(dice), dice_interval),
        (
            "boundary_noninferiority",
            "boundary F1 delta",
            _format_number(boundary),
            boundary_interval,
        ),
        ("clean_dice", "unavailable", "unavailable", "unavailable"),
        (
            "precision",
            "robust precision delta",
            _format_number(_receipt_number(receipt, "robust_deltas", "precision")),
            "unavailable",
        ),
        (
            "recall",
            "robust recall delta",
            _format_number(_receipt_number(receipt, "robust_deltas", "recall")),
            "unavailable",
        ),
        ("distance", "HD95 and ASSD deltas", distance_observed, "unavailable"),
        ("per_corruption", "unavailable", "unavailable", "unavailable"),
    )
    body = "\n".join(
        f"{_escape(gate_id)} & {_escape(endpoint)} &\n"
        f"unavailable & {_escape(observed)} & {_escape(interval)} & unavailable & unavailable & "
        f"unavailable & blocked & {_escape(source_reference)} \\\\"
        for gate_id, endpoint, observed, interval in rows
    )
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Loop206 gate audit derived only from the hash-bound closure receipt. Every reported interval is conditional on the three selected seeds. Missing receipt fields remain \texttt{{unavailable}} and force \texttt{{blocked}} rather than reconstructing a gate decision.}}
\label{{tab:loop206-gate-audit}}
\scriptsize
\def\sidprefix{{seed-}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llllllllll}}
\toprule
gate\_id & endpoint & threshold & observed & interval & \sidprefix206 & \sidprefix1206 & \sidprefix2206 & status & source \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""


def _legacy_table(registry: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in registry["observations"]
        if row["evidence_class"] == "legacy_patient_contaminated"
    ]
    body = "\n".join(
        f"{_escape(row['display_name'])} & {row['robust_dice']:.4f} & "
        f"{row['clean_dice']:.4f} & {row['robust_iou']:.4f} & {row['robust_bf1']:.4f} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Legacy Clean-v2 test-v2 values. Evidence class: \texttt{{legacy\_patient\_contaminated}}. Three patient IDs and 13 rows participate in cross-split contamination; these values are historical only.}}
\label{{tab:legacy-loop170}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Robust Dice & Clean Dice & Robust IoU & BF1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _scope_table() -> str:
    return r"""
\begin{table*}[t]
\centering
\caption{Evidence hierarchy used throughout this report.}
\label{tab:evidence-scope}
\small
\begin{tabular}{p{0.20\textwidth}p{0.20\textwidth}p{0.22\textwidth}p{0.22\textwidth}}
\toprule
Evidence class & Dataset/partition & Allowed use & Prohibited claim \\
\midrule
\path{protected_validation} & Clean-v3 validation & bounded model comparison & protected-test superiority \\
\path{train_screen} & Clean-v3 train-screen & controlled hypothesis rejection & generalization \\
\path{legacy_patient_contaminated} & Clean-v2 test-v2 & historical context & scientific ranking \\
\bottomrule
\end{tabular}
\end{table*}
"""


def build_tables(registry_path: str | Path, paper_dir: str | Path) -> dict[str, Path]:
    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    validate_registry(registry)
    release = paper_projection()
    if registry.get("release_manifest_sha256") != release["release_manifest_sha256"]:
        raise ValueError("evidence registry release manifest projection mismatch")
    root = Path(paper_dir)
    tables = root / "tables"
    manifest_path = root / "artifact_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid existing artifact manifest") from exc
        if not isinstance(loaded_manifest, dict):
            raise ValueError("existing artifact manifest must be an object")
        if (
            loaded_manifest.get("release_manifest_sha256")
            != release["release_manifest_sha256"]
        ):
            raise ValueError("artifact manifest release manifest projection mismatch")
        existing_manifest = loaded_manifest
        paper_pdf = existing_manifest.get("paper_pdf")
        if isinstance(paper_pdf, dict):
            built_input_digest = paper_pdf.get("built_paper_input_sha256")
            if not isinstance(built_input_digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", built_input_digest
            ) is None:
                raise ValueError(
                    "paper PDF input binding unavailable; promote before table rebuild"
                )
    table_texts = {
        "evidence_scope": _scope_table(),
        "clean_v3_validation": _clean_v3_table(registry),
        "loop206_ablation": _loop206_table(registry),
        "loop206_gate_audit": _loop206_gate_audit_table(registry, registry_file),
        "legacy_loop170": _legacy_table(registry),
    }
    table_overrides = {
        f"tables/{name}.tex": text.rstrip().encode("utf-8") + b"\n"
        for name, text in table_texts.items()
    }
    prospective_input_digest = _paper_input_sha256(root, table_overrides)
    manifest = {
        **existing_manifest,
        "schema_version": "imp.paper_artifacts.v1",
        "release_manifest_sha256": release["release_manifest_sha256"],
        "evidence_registry_path": registry_file.as_posix(),
        "evidence_registry_sha256": registry["registry_sha256"],
        "paper_input_sha256": prospective_input_digest,
        "tables": {
            name: {
                "path": f"tables/{name}.tex",
                "sha256": _sha256_bytes(table_overrides[f"tables/{name}.tex"]),
            }
            for name in sorted(table_texts)
        },
    }
    paper_pdf = manifest.get("paper_pdf")
    if isinstance(paper_pdf, dict):
        paper_pdf["status"] = "stale_uncompiled"
        manifest["paper_build"] = {
            "status": "building",
            "paper_input_sha256": prospective_input_digest,
        }
        _write_manifest(manifest_path, manifest)
    outputs = {
        name: _write(tables / f"{name}.tex", text)
        for name, text in table_texts.items()
    }
    current_input_digest = paper_input_sha256(root)
    manifest["paper_input_sha256"] = current_input_digest
    paper_pdf = manifest.get("paper_pdf")
    if isinstance(paper_pdf, dict):
        built_input_digest = paper_pdf.get("built_paper_input_sha256")
        if (
            built_input_digest == current_input_digest
            and paper_pdf.get("built_release_manifest_sha256")
            == release["release_manifest_sha256"]
        ):
            paper_pdf["status"] = "current"
        else:
            paper_pdf["status"] = "stale_uncompiled"
        manifest.pop("paper_build", None)
    _write_manifest(manifest_path, manifest)
    outputs["artifact_manifest"] = manifest_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-bound Clean-v3 paper tables")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--promote-paper-pdf", action="store_true")
    parser.add_argument("--expected-paper-input-sha256")
    parser.add_argument("--visual-review-passed", action="store_true")
    args = parser.parse_args()
    if args.promote_paper_pdf:
        if args.expected_paper_input_sha256 is None:
            parser.error("--expected-paper-input-sha256 is required for promotion")
        manifest = promote_paper_pdf(
            args.paper_dir,
            expected_paper_input_sha256=args.expected_paper_input_sha256,
            visual_review_passed=args.visual_review_passed,
        )
        print(
            "paper_pdf_status=current "
            f"paper_input_sha256={args.expected_paper_input_sha256} manifest={manifest}"
        )
        return
    if args.registry is None:
        parser.error("--registry is required for table generation")
    outputs = build_tables(args.registry, args.paper_dir)
    print(f"tables_status=valid count={len(outputs) - 1} manifest={outputs['artifact_manifest']}")


if __name__ == "__main__":
    main()
