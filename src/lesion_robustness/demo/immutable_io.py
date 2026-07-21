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

    def decode_rgb(self):
        import numpy as np
        from PIL import Image
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
            with Image.open(self.open()) as image:
                if int(image.width) * int(image.height) > 64_000_000:
                    raise ValueError("verified source image exceeds trusted pixel cap")
                return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    @staticmethod
    def decoded_rgb_sha256(image) -> str:
        import numpy as np

        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("decoded RGB hash requires an RGB uint8 array")
        contiguous = np.ascontiguousarray(rgb)
        payload = (
            f"{contiguous.shape[0]}x{contiguous.shape[1]}x3|".encode("ascii")
            + contiguous.tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    def decode_binary_mask(self):
        import numpy as np
        from PIL import Image
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
            with Image.open(self.open()) as image:
                if int(image.width) * int(image.height) > 64_000_000:
                    raise ValueError("verified source mask exceeds trusted pixel cap")
                gray = np.array(image.convert("L"), dtype=np.uint8, copy=True)
        return (gray > 127).astype(np.uint8)

    @staticmethod
    def decoded_binary_mask_sha256(mask) -> str:
        import numpy as np

        binary = np.asarray(mask)
        if binary.ndim != 2 or binary.dtype != np.uint8 or not np.isin(binary, (0, 1)).all():
            raise ValueError("decoded mask hash requires a binary uint8 array")
        contiguous = np.ascontiguousarray(binary)
        payload = (
            f"{contiguous.shape[0]}x{contiguous.shape[1]}|".encode("ascii")
            + contiguous.tobytes()
        )
        return hashlib.sha256(payload).hexdigest()
