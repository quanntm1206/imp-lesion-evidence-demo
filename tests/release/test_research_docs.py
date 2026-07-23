from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_public_tree_omits_agent_artifacts_and_legacy_configs() -> None:
    internal_paths = (
        ".superpowers/sdd/completion-task1-report.md",
        "docs/superpowers/plans/2026-07-20-evidence-first-paper-demo-rescue.md",
        "docs/superpowers/plans/2026-07-21-dual-live-demo.md",
        "docs/superpowers/plans/2026-07-21-interactive-research-deck.md",
        "docs/superpowers/specs/2026-07-20-evidence-first-paper-demo-rescue-design.md",
        "docs/superpowers/specs/2026-07-21-dual-live-demo-design.md",
        "docs/superpowers/specs/2026-07-21-interactive-research-deck-design.md",
        "docs/presentation/2026-07-23-professor-p-fast-lane-audit.md",
        "docs/presentation/professor-audit-report.md",
        "docs/presentation/presenter-s-transcript.md",
        "docs/presentation/defense-question-bank.md",
        "reports/paper_revision/manuscript_readiness_audit.md",
    )
    assert [path for path in internal_paths if (ROOT / path).is_file()] == []

    public_configs = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs").rglob("*.yaml")
    }
    assert public_configs == {
        "configs/demo/loop206_live.yaml",
        "configs/loop206/l206_control_train_screen_pilot20.yaml",
    }

    gitignore = _read(".gitignore")
    assert "/.superpowers/" in gitignore
    assert "/docs/superpowers/" in gitignore


def test_citation_identifies_paper_and_both_authors() -> None:
    citation = yaml.safe_load(_read("CITATION.cff"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["title"] == (
        "Evidence-Bounded Comparison of MiT-B3 U-Net and nnU-Net "
        "for Skin Lesion Segmentation"
    )
    assert citation["repository-code"] == (
        "https://github.com/quanntm1206/imp-lesion-evidence-demo"
    )
    assert [author["given-names"] for author in citation["authors"]] == [
        "Minh Quân",
        "Đức Lân",
    ]
    assert [author["family-names"] for author in citation["authors"]] == [
        "Nguyễn Trần",
        "Nguyễn",
    ]


def test_readme_states_release_boundary_and_four_evidence_lanes() -> None:
    readme = _read("README.md")

    required = (
        "reproducibility scaffold",
        "Historical Paper RQ1",
        "Loop206 train-screen ablation",
        "Live reconstructed runtime",
        "Prospective RQ1-v2",
        "pending/unverified",
        "IMP MiT-B3 U-Net",
        "reconstructed nnU-Net",
        "outside GitHub",
    )
    assert all(term in readme for term in required)
    assert "not a clone-runnable training release" in readme


def test_reproducibility_doc_distinguishes_validation_from_reproduction() -> None:
    document = _read("docs/reproducibility.md")

    required = (
        "Registry-only verification",
        "Strict local audit",
        "not independently reconstructable",
        "Private artifact transfer",
        "Exit code 2",
        "six jobs",
        "pending/unverified",
    )
    assert all(term in document for term in required)


def test_documented_paper_audits_supply_required_receipts() -> None:
    document = _read("docs/reproducibility.md")

    assert document.count("audit_clean_v3_paper.py") == 2
    assert document.count("--receipt") == 2


def test_model_card_bounds_intended_use_and_runtime_claims() -> None:
    document = _read("docs/model-card.md")

    required = (
        "IMP MiT-B3 U-Net",
        "MiT-B3 encoder",
        "U-Net decoder",
        "reconstructed nnU-Net",
        "illustrative",
        "not a diagnostic or clinical system",
        "original-runtime equivalence",
        "subgroup fairness",
    )
    assert all(term in document for term in required)


def test_data_card_separates_historical_and_prospective_admission() -> None:
    document = _read("docs/data-card.md")

    required = (
        "ISIC 2016, 2017, and 2018",
        "2,008",
        "431",
        "430",
        "historical recorded audit",
        "prospective RQ1-v2 admission",
        "not independently reconstructable",
        "test-v3",
        "PH2",
        "patient-level",
    )
    assert all(term in document for term in required)


def test_release_docs_reject_affirmative_overclaims_and_private_urls() -> None:
    documents = "\n".join(
        _read(relative)
        for relative in (
            "README.md",
            "CITATION.cff",
            "docs/reproducibility.md",
            "docs/model-card.md",
            "docs/data-card.md",
        )
    )
    affirmative = re.compile(
        r"(?i)(?:"
        r"\b(?:is|are|shows?|demonstrates?|establishes?|achieves?|supports?)"
        r"\s+(?:a\s+)?(?:clinical|state[- ]of[- ]the[- ]art|SOTA|"
        r"statistical superiority|superiority|original-runtime equivalence)"
        r"|\b(?:clinically validated|clinical-grade|superior to|"
        r"equivalent to (?:the )?original)\b)"
    )

    paragraphs = (
        " ".join(paragraph.splitlines())
        for paragraph in re.split(r"\n\s*\n", documents)
    )
    for paragraph in paragraphs:
        for match in affirmative.finditer(paragraph):
            prefix = paragraph[max(0, match.start() - 160) : match.start()]
            if re.search(
                r"(?i)\b(?:no|not|never|without|cannot|does not|do not)\b",
                prefix,
            ) is None:
                raise AssertionError(paragraph)
    assert not re.search(r"(?i)(?:trycloudflare\.com|\.trycloudflare\.com)", documents)
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", documents)
    windows_user = r"\\" + r"Users\\[^\\\s]+"
    assert not re.search(
        r"(?i)(?:/Users/|/home/[^/\s]+|" + windows_user + ")", documents
    )


def test_release_docs_pin_a_tunnel_policy_without_a_tunnel_url() -> None:
    readme = _read("README.md")

    assert "No public tunnel URL is pinned" in readme
    assert "Quick Tunnel" in readme
    assert "unauthenticated" in readme


def test_documented_entrypoint_paths_exist() -> None:
    paths = (
        "scripts/bootstrap_windows.ps1",
        "scripts/demo/run_sidecar.ps1",
        "scripts/demo/run_demo.ps1",
        "scripts/paper/audit_clean_v3_paper.py",
        "scripts/research/train_rq1_v2.py",
        "scripts/research/evaluate_rq1_v2.py",
        "scripts/research/reproduce_paper_results.ps1",
        "experiments/rq1_v2/protocol.json",
        "docs/runbooks/two-machine-delivery.md",
    )

    assert all((ROOT / relative).is_file() for relative in paths)


def test_powershell_command_blocks_reference_existing_scripts() -> None:
    documents = "\n".join(
        (_read("README.md"), _read("docs/reproducibility.md"))
    )
    command_blocks = re.findall(
        r"```powershell\s*(.*?)```", documents, flags=re.DOTALL
    )
    script_tokens = {
        token.replace("\\", "/")
        for block in command_blocks
        for token in re.findall(r"scripts[/\\][A-Za-z0-9_./\\-]+", block)
    }
    audit_blocks = [
        block for block in command_blocks if "audit_clean_v3_paper.py" in block
    ]

    assert script_tokens
    assert all((ROOT / token).is_file() for token in script_tokens)
    assert len(audit_blocks) == 2
    assert all(
        all(flag in block for flag in ("--paper", "--registry", "--receipt"))
        for block in audit_blocks
    )


def test_readme_has_copyable_contract_only_rq1_v2_commands() -> None:
    readme = _read("README.md")
    logical_lines = re.sub(r"`\r?\n\s*", " ", readme).splitlines()

    for script in ("train_rq1_v2.py", "evaluate_rq1_v2.py"):
        commands = [line for line in logical_lines if script in line]
        assert any("--dry-run" in command for command in commands)
        assert any("--preflight-only" in command for command in commands)

    runner_commands = [
        line for line in logical_lines if "reproduce_paper_results.ps1" in line
    ]
    assert any("-DryRun" in command for command in runner_commands)
    assert any("-PreflightOnly" in command for command in runner_commands)
    assert "contract-only scaffold" in readme
    assert "pending/unverified" in readme


def test_teacher_demo_guide_is_linked_and_self_deployable() -> None:
    guide_path = ROOT / "DEMO_DEPLOYMENT_GUIDE.md"
    assert guide_path.is_file()

    guide = guide_path.read_text(encoding="utf-8")
    assert "DEMO_DEPLOYMENT_GUIDE.md" in _read("README.md")
    assert "DEMO_DEPLOYMENT_GUIDE.md" in _read("demo/README.md")

    required = (
        "Không thể chạy inference thật chỉ từ GitHub",
        "IMP_LOOP206_CONTROL_CHECKPOINT",
        "IMP_LOOP206_CANDIDATE_CHECKPOINT",
        "IMP_LOOP206_DATA_ROOT",
        "demo_runtime/loop206_dataset_index.json",
        "demo_runtime/nnunet/recovered-container-final2",
        "sha256-manifest.json",
        "recovery_receipt.json",
        "pilot_cache_v2_candidate",
        "pilot_cache_v2_zero_control",
        "preflight=passed",
        "dual_smoke=passed",
        "127.0.0.1:7860",
        "127.0.0.1:7862",
        "Cloudflare",
        "Troubleshooting",
    )
    assert all(term in guide for term in required)

    ordered_commands = (
        "scripts/demo/run_sidecar.ps1 -CheckOnly -PreserveMode -RunId $RunId",
        "scripts/demo/run_sidecar.ps1 -PreserveMode -RunId $RunId",
        "scripts/demo/run_demo.ps1 -CheckOnly -PublicTunnelMode -PreserveMode -RunId $RunId -PythonExe $PythonExe",
        "scripts/demo/run_demo.ps1 -PublicTunnelMode -PreserveMode -RunId $RunId -PythonExe $PythonExe",
        "scripts/demo/run_tunnel.ps1 -PreserveMode -RunId $RunId",
        "scripts/demo/stop_demo.ps1 -PreserveMode -RunId $RunId",
    )
    positions = [guide.index(command) for command in ordered_commands]
    assert positions == sorted(positions)

    current_image = (
        "sha256:86bd77c03c3918e3638565e29417cdf4360b499a0813fbc425dc36645f026f2d"
    )
    stale_image = (
        "sha256:4e4c3be63c95834f36fdcd1b0c66ec60203f3da12239eafb603d3e815d5f89ae"
    )
    assert current_image in guide
    assert current_image in _read("docs/runbooks/demo-operations.md")
    assert stale_image not in guide
    assert stale_image not in _read("docs/runbooks/demo-operations.md")
    assert not re.search(r"(?i)trycloudflare\.com", guide)
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", guide)
    assert "git switch main" in guide
    assert "IMP_DEMO_ARTIFACT_MANIFEST" in guide
    assert "GetFullPath" in guide
    assert "StartsWith" in guide
    assert "Compare-Object" in guide
    assert ".Replace('\\', '/')" in guide
    assert "Get-ChildItem -LiteralPath $ArtifactRoot -Force -Recurse -File" in guide
    assert "git switch --detach submission-2026-07-23-v3" in guide
    assert "git switch --detach submission-2026-07-23-v2" not in guide
    assert "repository-overlay" not in guide
    assert "@('.artifacts', 'demo_runtime')" in guide
    assert "Copy-Item -LiteralPath $Source -Destination $Destination -Recurse" in guide
