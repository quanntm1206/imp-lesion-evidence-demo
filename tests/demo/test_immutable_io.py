from __future__ import annotations

from pathlib import Path

from lesion_robustness.demo.immutable_io import ImmutableSnapshot


def test_large_snapshot_streams_to_owned_file_and_survives_source_replacement(
    tmp_path: Path,
) -> None:
    expected = bytes(range(251)) * 8
    source = tmp_path / "artifact.bin"
    source.write_bytes(expected)

    snapshot = ImmutableSnapshot.read(source, chunk_size=17, spool_limit=64)
    source.write_bytes(b"replacement")

    assert snapshot.size == len(expected)
    assert snapshot.is_file_backed
    with snapshot.open() as handle:
        assert handle.read() == expected
    with snapshot.open() as handle:
        assert handle.read() == expected
