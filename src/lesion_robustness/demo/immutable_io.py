"""Single-open immutable snapshots for evidence-bound runtime artifacts."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
from typing import Any, BinaryIO


DEFAULT_SPOOL_LIMIT = 8 << 20


class _SnapshotReader:
    """Seekable read-only view that cannot close or mutate the owned snapshot."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle

    def __enter__(self) -> "_SnapshotReader":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def close(self) -> None:
        return None

    def writable(self) -> bool:
        return False

    def write(self, _data: object) -> int:
        raise io.UnsupportedOperation("immutable snapshot is read-only")

    def truncate(self, _size: int | None = None) -> int:
        raise io.UnsupportedOperation("immutable snapshot is read-only")


class ImmutableSnapshot:
    """Owned seekable copy whose digest describes the exact readable bytes."""

    def __init__(self, handle: BinaryIO, sha256: str, size: int) -> None:
        self._handle = handle
        self.sha256 = sha256
        self.size = size

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        chunk_size: int = 1 << 20,
        spool_limit: int = DEFAULT_SPOOL_LIMIT,
    ) -> "ImmutableSnapshot":
        if chunk_size < 1 or spool_limit < 1:
            raise ValueError("snapshot limits must be positive")
        digest = hashlib.sha256()
        size = 0
        owned = tempfile.SpooledTemporaryFile(max_size=spool_limit, mode="w+b")
        try:
            with Path(path).open("rb") as source:
                while chunk := source.read(chunk_size):
                    digest.update(chunk)
                    owned.write(chunk)
                    size += len(chunk)
            owned.flush()
            owned.seek(0)
            return cls(owned, digest.hexdigest(), size)
        except BaseException:
            owned.close()
            raise

    @classmethod
    def from_bytes(
        cls, data: bytes, *, spool_limit: int = DEFAULT_SPOOL_LIMIT
    ) -> "ImmutableSnapshot":
        if spool_limit < 1:
            raise ValueError("snapshot limits must be positive")
        owned = tempfile.SpooledTemporaryFile(max_size=spool_limit, mode="w+b")
        owned.write(data)
        owned.flush()
        owned.seek(0)
        return cls(owned, hashlib.sha256(data).hexdigest(), len(data))

    @property
    def is_file_backed(self) -> bool:
        return bool(getattr(self._handle, "_rolled", True))

    def open(self) -> _SnapshotReader:
        self._handle.seek(0)
        return _SnapshotReader(self._handle)

    def read_bytes(self, *, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and self.size > max_bytes:
            raise ValueError("snapshot exceeds in-memory read limit")
        with self.open() as handle:
            return handle.read()

    def text(self, encoding: str) -> str:
        return self.read_bytes().decode(encoding)
