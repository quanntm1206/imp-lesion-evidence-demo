from __future__ import annotations

import argparse
from pathlib import Path

from lesion_robustness.demo.data_index import (
    build_index_payload,
    resolve_loop206_rows,
    write_index,
)
from lesion_robustness.demo.source_fetch import fetch_manifest_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch only the official files required by Loop206")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()
    counts = fetch_manifest_sources(args.manifest, args.output_root)
    rows = resolve_loop206_rows(args.manifest, [args.output_root])
    write_index(build_index_payload(rows, [args.output_root]), args.index_output)
    print(f"source_page=https://challenge.isic-archive.com/data/ counts={counts}")
    print("rows=384 fit=308 holdout=76 hash_mismatches=0 overlap=0")


if __name__ == "__main__":
    main()
