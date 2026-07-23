"""Freeze synthetic RQ1-v2 condition reference vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import PIL

from lesion_robustness.research.rq1_protocol import build_condition_panel


def _fixture_bytes() -> bytes:
    return (np.arange(32 * 32 * 3, dtype=np.uint32) % 256).astype(np.uint8).tobytes()


def freeze(protocol_path: Path, input_path: Path, output_path: Path) -> dict[str, object]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("ascii"))
    expected_input = _fixture_bytes()
    if not input_path.exists():
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(expected_input)
    input_bytes = input_path.read_bytes()
    if input_bytes != expected_input:
        raise ValueError("condition fixture must be uint8[32,32,3] arange modulo 256")
    rgb = np.frombuffer(input_bytes, dtype=np.uint8).reshape(32, 32, 3)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol_with_hash = dict(protocol)
    protocol_with_hash["protocol_sha256"] = protocol_sha256
    panel = build_condition_panel(
        rgb,
        {"group_key": "RQ1v2-fixture-group", "sample_id": "RQ1v2-fixture-sample"},
        protocol_with_hash,
    )
    payload: dict[str, object] = {
        "schema_version": "imp.rq1_v2.condition_golden.v1",
        "protocol_sha256": protocol_sha256,
        "fixture_group_key": "RQ1v2-fixture-group",
        "fixture_sample_id": "RQ1v2-fixture-sample",
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "condition_uint64_seeds": dict(panel.seeds),
        "condition_rgb_sha256": dict(panel.hashes),
        "ordered_panel_sha256": panel.ordered_panel_sha256,
        "dependency_versions": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
        },
    }
    # The checked-in golden was frozen with Windows text newlines; preserve those
    # bytes so reruns can verify immutability without rewriting the artifact.
    serialized = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").replace("\n", "\r\n").encode("ascii")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            existing = output_path.read_bytes()
        except OSError as exc:
            raise ValueError("existing golden could not be read") from exc
        if existing != serialized:
            raise ValueError("existing golden drift; refusing overwrite")
    else:
        output_path.write_bytes(serialized)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.protocol, args.input, args.output)
    for name, seed in payload["condition_uint64_seeds"].items():
        print(f"{name} seed={seed} rgb_sha256={payload['condition_rgb_sha256'][name]}")
    print(f"ordered_panel_sha256={payload['ordered_panel_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
