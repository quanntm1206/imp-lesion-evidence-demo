from pathlib import Path
import re

QUESTION_BANK = Path("docs/presentation/defense-question-bank.md")
HISTORICAL_AUDITS = (
    Path("docs/presentation/professor-audit-report.md"),
    Path("docs/presentation/2026-07-23-professor-p-fast-lane-audit.md"),
)


def test_acceptance_identity_is_stable():
    paths = [
        Path("README.md"), Path("demo/README.md"),
        Path("docs/runbooks/demo-operations.md"),
        Path("docs/runbooks/two-machine-delivery.md"),
        QUESTION_BANK, Path("docs/presentation/presenter-s-transcript.md"),
        Path("paper/clean_v3_loop206"), Path("presentation/interactive/content.json"),
        Path("release/imp_release_manifest.json"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    assert "imp.dual_live.e2e.v1" in text
    assert "demo_runtime/acceptance/imp.dual_live.e2e.v1/$RunId/acceptance.json" in text
    assert not re.search(r"(?<!\.superpowers/sdd/)task-[0-9]+", text)


def test_historical_professor_audits_are_not_current_release_evidence() -> None:
    for path in HISTORICAL_AUDITS:
        text = path.read_text(encoding="utf-8")
        header = "\n".join(text.splitlines()[:6])
        assert header.startswith("# HISTORICAL / SUPERSEDED")
        assert "must not be treated as current release evidence" in header
        assert "Authoritative current state" in header
        assert "status=current" in header
        assert "package_state=complete" in header
        assert "17 slides" in header
        assert "435606d5adc296be57405c65a9c725af3dff96c15f9aabf7ac0924d06387a264" in header
        assert "unverified/blocked" in header
        assert "inspection snapshot" in header
