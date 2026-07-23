"""Fail-closed contract entry point for prospective RQ1-v2 evaluation."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rq1_v2 import build_parser, run_contract  # noqa: E402


def main() -> int:
    parser = build_parser("Validate the contract-only RQ1-v2 evaluation surface", operation="evaluate")
    return run_contract(parser.parse_args(), operation="evaluate")


if __name__ == "__main__":
    raise SystemExit(main())
