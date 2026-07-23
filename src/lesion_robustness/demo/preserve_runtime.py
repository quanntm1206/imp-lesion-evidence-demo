"""Append-only runtime journals for demo records that must be retained."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, Mapping


class PreserveJournal:
    """Write immutable records; current state is an in-memory logical pointer."""

    def __init__(self, runtime_root: Path, *, run_id: str | None = None) -> None:
        self._runtime_root = Path(runtime_root).resolve(strict=False)
        self._run_id = self.validate_run_id(run_id or self._new_run_id())
        self.validate_path_budget(self._runtime_root, self._run_id)
        preserved_path = self._runtime_root / "preserved"
        self._ensure_directory(preserved_path, "preserved runtime root")
        preserved_root = preserved_path.resolve(strict=True)
        run_path = preserved_root / self._run_id
        self._ensure_directory(run_path, "preserved run")
        self._root = run_path.resolve(strict=True)
        try:
            self._root.relative_to(preserved_root)
        except ValueError as exc:
            raise ValueError("run_id escapes the preserved runtime root") from exc
        self._current_component: str | None = None
        self._latest: dict[str, Path] = {}

    @staticmethod
    def _new_run_id() -> str:
        return secrets.token_hex(16)

    @staticmethod
    def validate_run_id(value: str) -> str:
        if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", value) is None:
            raise ValueError("run_id is unsafe")
        return value

    @staticmethod
    def validate_path_budget(
        runtime_root: Path, run_id: str, *, windows: bool | None = None
    ) -> None:
        safe_run_id = PreserveJournal.validate_run_id(run_id)
        projected = (
            Path(runtime_root).resolve(strict=False)
            / "preserved"
            / safe_run_id
            / "receipt"
            / ("started-" + "0" * 32 + ".json")
        )
        applies = os.name == "nt" if windows is None else windows
        if applies and len(str(projected)) >= 248:
            raise ValueError("preserved runtime path budget exceeds safe Windows limit")

    @staticmethod
    def _ensure_directory(path: Path, label: str) -> None:
        if path.exists():
            attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
            reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            is_junction = bool(
                getattr(os.path, "isjunction", lambda _path: False)(path)
            )
            if (
                not path.is_dir()
                or path.is_symlink()
                or is_junction
                or bool(attributes & reparse_flag)
            ):
                raise ValueError(f"{label} must be a real directory")
            return
        path.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _component(value: str) -> str:
        if not isinstance(value, str) or not value or value in {".", ".."}:
            raise ValueError("journal component must be nonempty")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
            raise ValueError("journal component is unsafe")
        return value

    @staticmethod
    def _event(value: str) -> str:
        return PreserveJournal._component(value)

    def _write(self, component: str, event: str, payload: Mapping[str, Any]) -> Path:
        directory = self._root / component
        self._ensure_directory(directory, "journal component")
        directory = directory.resolve(strict=True)
        try:
            directory.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("journal component escapes the preserved run") from exc
        record = {
            "component": component,
            "event": event,
            "recorded_at_ns": time.time_ns(),
            "payload": dict(payload),
        }
        encoded = json.dumps(
            record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        ) + "\n"
        while True:
            path = directory / f"{event}-{secrets.token_hex(16)}.json"
            try:
                with path.open("x", encoding="ascii", newline="\n") as handle:
                    handle.write(encoded)
            except FileExistsError:
                continue
            self._latest[component] = path
            return path

    def start(self, component: str, identity: Mapping[str, Any]) -> Path:
        component = self._component(component)
        self._current_component = component
        return self._write(component, "started", identity)

    def append(self, event: str, payload: Mapping[str, Any] | None = None) -> Path:
        if self._current_component is None:
            raise RuntimeError("journal component has not been started")
        return self._write(self._current_component, self._event(event), payload or {})

    def latest(self, component: str) -> Path | None:
        return self._latest.get(self._component(component))
