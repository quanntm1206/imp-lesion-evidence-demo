from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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
    rows = [
        _observation(registry, "L191-C0-clean-v3-IMP-control"),
        _observation(registry, "L192-nnUNet-v2-raw-100ep"),
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
    root = Path(paper_dir)
    tables = root / "tables"
    outputs = {
        "evidence_scope": _write(tables / "evidence_scope.tex", _scope_table()),
        "clean_v3_validation": _write(
            tables / "clean_v3_validation.tex", _clean_v3_table(registry)
        ),
        "loop206_ablation": _write(tables / "loop206_ablation.tex", _loop206_table(registry)),
        "legacy_loop170": _write(tables / "legacy_loop170.tex", _legacy_table(registry)),
    }
    manifest_path = root / "artifact_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid existing artifact manifest") from exc
        if not isinstance(loaded_manifest, dict):
            raise ValueError("existing artifact manifest must be an object")
        existing_manifest = loaded_manifest

    manifest = {
        **existing_manifest,
        "schema_version": "imp.paper_artifacts.v1",
        "evidence_registry_path": registry_file.as_posix(),
        "evidence_registry_sha256": registry["registry_sha256"],
        "tables": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for name, path in sorted(outputs.items())
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    outputs["artifact_manifest"] = manifest_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-bound Clean-v3 paper tables")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = build_tables(args.registry, args.paper_dir)
    print(f"tables_status=valid count={len(outputs) - 1} manifest={outputs['artifact_manifest']}")


if __name__ == "__main__":
    main()
