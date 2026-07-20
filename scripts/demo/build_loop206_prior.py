from __future__ import annotations

import argparse
from pathlib import Path

from lesion_robustness.demo.loop206_prior import build_prior_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the exact Loop206 deployment prior")
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--dataset-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    receipt = build_prior_artifact(
        dataset_index=args.dataset_index,
        candidate_manifest=args.candidate_cache,
        output=args.output,
        receipt=args.receipt,
        n_jobs=args.n_jobs,
        dataset_roots=args.dataset_root,
    )
    parity = receipt["parity"]
    print(
        f"fit_groups={receipt['fit_groups']} "
        f"holdout_parity={parity['contour_byte_matches']}/{parity['expected']} "
        f"parity_passed={str(parity['parity_passed']).lower()}"
    )


if __name__ == "__main__":
    main()
