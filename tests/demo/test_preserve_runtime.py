from __future__ import annotations

from pathlib import Path

import pytest

from lesion_robustness.demo.preserve_runtime import PreserveJournal


def test_task7_timestamp_and_max_run_ids_fit_current_worktree_budget() -> None:
    runtime_root = Path(__file__).resolve().parents[2] / "demo_runtime"
    for run_id in ("20260722t1234561234567z", "a" * 128):
        PreserveJournal.validate_path_budget(runtime_root, run_id)


def test_preserve_journal_rejects_over_budget_windows_projection(tmp_path: Path) -> None:
    runtime_root = tmp_path / ("x" * 170) / "demo_runtime"
    with pytest.raises(ValueError, match="path budget"):
        PreserveJournal.validate_path_budget(runtime_root, "a" * 128, windows=True)


def test_journal_records_are_unique_immutable_and_latest_is_logical(tmp_path: Path) -> None:
    journal = PreserveJournal(tmp_path)

    started = journal.start("gradio", {"generation": 1})
    stopped = journal.append("stopped", {"generation": 1})

    assert started.is_file() and stopped.is_file()
    assert started != stopped
    assert stopped.parent.parent.parent == tmp_path / "preserved"
    assert journal.latest("gradio") == stopped
    assert started.read_text(encoding="ascii") != stopped.read_text(encoding="ascii")


def test_logical_invalidation_never_unlinks_historical_records(
    tmp_path: Path, monkeypatch
) -> None:
    journal = PreserveJournal(tmp_path)
    receipt = journal.start("receipt", {"generation": 1})

    monkeypatch.setattr(Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("delete")))
    invalidated = journal.append("invalidated", {"generation": 2})

    assert receipt.is_file()
    assert invalidated.is_file()
    assert journal.latest("receipt") == invalidated


@pytest.mark.parametrize(
    "run_id",
    ("../escape", "..", ".", "/rooted", "C:\\rooted", "has/slash", "has\\slash"),
)
def test_run_id_rejects_escape_and_rooted_values(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        PreserveJournal(tmp_path, run_id=run_id)


def test_safe_explicit_run_id_stays_contained(tmp_path: Path) -> None:
    journal = PreserveJournal(tmp_path, run_id="run_20260722-a")

    record = journal.start("gradio", {"generation": 1})

    assert record.resolve().is_relative_to((tmp_path / "preserved").resolve())


def test_journal_joins_existing_validated_run_without_reusing_event_file(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "preserved/shared-run/gradio/owner-history.json"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical", encoding="ascii")

    journal = PreserveJournal(tmp_path, run_id="shared-run")
    receipt = journal.start("receipt", {"generation": 1})

    assert historical.read_text(encoding="ascii") == "historical"
    assert receipt.is_file()
    assert receipt.parent == tmp_path / "preserved/shared-run/receipt"


def test_existing_run_path_must_be_a_real_directory(tmp_path: Path) -> None:
    run_path = tmp_path / "preserved/shared-run"
    run_path.parent.mkdir(parents=True)
    run_path.write_text("not a directory", encoding="ascii")

    with pytest.raises((OSError, ValueError)):
        PreserveJournal(tmp_path, run_id="shared-run")
