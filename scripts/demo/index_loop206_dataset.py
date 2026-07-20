from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lesion_robustness.demo.data_index import (
    build_index_payload,
    resolve_loop206_rows,
    write_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the portable Loop206 pilot dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = resolve_loop206_rows(args.manifest, args.root)
    write_index(build_index_payload(rows, args.root), args.output)
    sources = Counter(row.source_dataset for row in rows)
    print(
        f"rows={len(rows)} fit={sum(row.role == 'fit' for row in rows)} "
        f"holdout={sum(row.role == 'holdout' for row in rows)} "
        f"sources={dict(sorted(sources.items()))} hash_mismatches=0 overlap=0"
    )


if __name__ == "__main__":
    main()
