from __future__ import annotations

import argparse
from pathlib import Path

from lesion_robustness.research.rq1_data import (
    audit_data,
    canonical_json_bytes,
    freeze_protocol_identities,
    load_protocol,
    protocol_payload,
    read_authorized_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit authorized RQ1-v2 train/validation data")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("RQ1-v2 integrity report is immutable and already exists")

    protocol = load_protocol(args.protocol)
    train = read_authorized_rows(args.index, "train", protocol)
    validation = read_authorized_rows(args.index, "validation", protocol)
    report = audit_data((*train, *validation), protocol)
    frozen = freeze_protocol_identities(protocol, report)
    args.protocol.write_bytes(canonical_json_bytes(protocol_payload(frozen)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(report.to_dict()))
    print(
        f"train={report.train_count} validation={report.validation_count} "
        f"test_opened={report.test_v3_open_count} cross_group={report.cross_split_groups} "
        f"cross_exact={report.cross_split_exact_rgb} cross_near={report.cross_split_near_rgb}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
