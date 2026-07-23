from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.release.audit_public_surface import audit, main


def test_public_audit_accepts_canonical_source_and_blocked_manifest(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("def predict(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "results" / "rq1_v2").mkdir(parents=True)
    (tmp_path / "results" / "rq1_v2" / "result_manifest.json").write_text(
        '{"status":"pending/unverified","p1_status":"not_promoted","metrics":[],"completed_jobs":0,"required_jobs":6}\n',
        encoding="utf-8",
    )
    result = audit(tmp_path, mode="working-tree")
    assert not result.forbidden, result.findings


def test_public_audit_rejects_weights_and_tunnel_url(tmp_path: Path) -> None:
    weight = tmp_path / "weights" / "best.pt"
    weight.parent.mkdir()
    weight.write_bytes(b"x")
    tunnel_host = "try" + "cloudflare.com"
    (tmp_path / "README.md").write_text(
        "https://example." + tunnel_host, encoding="utf-8"
    )
    result = audit(tmp_path, mode="working-tree")
    assert result.forbidden
    assert any("weights" in finding or "trycloudflare" in finding for finding in result.findings)


def test_public_audit_rejects_private_paths_archives_and_raw_receipts(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "local " + "C:" + chr(92) + "Users" + chr(92) + "Admin" + chr(92) + "secret" + chr(92) + "model.pt", encoding="utf-8"
    )
    (tmp_path / "backup.zip").write_bytes(b"PK\x03\x04archive")
    (tmp_path / "raw_receipts").mkdir()
    (tmp_path / "raw_receipts" / "run.json").write_text("{}", encoding="ascii")
    result = audit(tmp_path, mode="working-tree")
    assert result.forbidden
    assert len(result.findings) >= 3


def test_public_audit_rejects_historical_qualitative_receipt_bundle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper" / "clean_v3_loop206" / "figures"
    path.mkdir(parents=True)
    bundle = path / ("qualitative_demo_" + "receipts.json")
    bundle.write_text(
        json.dumps(
            {
                "schema_version": "loop206.qualitative_" + "receipts.v1",
                "receipts": [],
            }
        ),
        encoding="ascii",
    )

    result = audit(tmp_path, mode="working-tree")

    assert result.forbidden
    assert any("raw receipt" in finding for finding in result.findings)


@pytest.mark.parametrize(
    "name",
    [
        "qualitative-demo-receipt-bundle.json",
        "demo.receipts.snapshot.json",
        "run_raw-receipt.json",
    ],
)
def test_public_audit_rejects_receipt_name_variants(
    tmp_path: Path, name: str,
) -> None:
    (tmp_path / name).write_text("{}", encoding="ascii")

    result = audit(tmp_path, mode="working-tree")

    assert result.forbidden
    assert any("raw receipt" in finding for finding in result.findings)


def test_public_audit_rejects_json_receipts_array_in_safe_filename(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"receipts": [{"sample": "private"}]}), encoding="ascii"
    )

    result = audit(tmp_path, mode="working-tree")

    assert result.forbidden
    assert any("receipt array" in finding for finding in result.findings)


@pytest.mark.parametrize(
    "schema_id",
    [
        "loop206.demo." + "receipt.v1",
        "loop206.qualitative_" + "receipts.v1",
    ],
)
def test_public_audit_rejects_receipt_schema_ids(
    tmp_path: Path, schema_id: str,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"schema_version": schema_id}), encoding="ascii"
    )

    result = audit(tmp_path, mode="working-tree")

    assert result.forbidden
    assert any("receipt schema" in finding for finding in result.findings)


def test_public_audit_accepts_compact_summary_at_historical_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper" / "clean_v3_loop206" / "figures"
    path.mkdir(parents=True)
    summary = path / ("qualitative_demo_" + "receipts.json")
    summary.write_text(
        json.dumps(
            {
                "aggregate_mask_bindings_sha256": "a" * 64,
                "artifact_role": "derived_public_aggregate_provenance",
                "authorized_sample_count": 3,
                "evidence_class": "train_screen / exact_fixed_cache / historical_cache_provenance_drift",
                "evidence_registry_sha256": "b" * 64,
                "external_runtime_bundle_sha256": "c" * 64,
                "panel_caption": "illustrative; not protected-test evidence",
                "provenance_manifest_sha256": "d" * 64,
                "release_manifest_sha256": "e" * 64,
                "schema_version": "loop206.qualitative_public_summary.v1",
                "source_record_count": 3,
            }
        ),
        encoding="ascii",
    )

    result = audit(tmp_path, mode="working-tree")

    assert not result.forbidden, result.findings


def test_public_audit_rejects_multi_suffix_archives_and_bare_tunnel_hosts(
    tmp_path: Path,
) -> None:
    (tmp_path / "backup.tar.gz").write_bytes(b"archive")
    tunnel_host = ".".join(("demo", "try" + "cloudflare", "com"))
    (tmp_path / "README.md").write_text(
        f"host: {tunnel_host}\n", encoding="ascii"
    )
    result = audit(tmp_path, mode="working-tree")
    assert result.forbidden
    assert any("binary archive" in finding for finding in result.findings)
    assert any("tunnel URL" in finding for finding in result.findings)


@pytest.mark.parametrize(
    "private_literal",
    [
        "/".join(("home", "admin_" + "mugen")),
        "/" + "/".join(("home", "admin_" + "mugen")),
        "/".join(("home", "admin_" + "mugen")) + "/",
    ],
)
def test_public_audit_rejects_exact_historic_private_literals(
    tmp_path: Path, private_literal: str,
) -> None:
    (tmp_path / "README.md").write_text(
        f"old path: {private_literal}\n", encoding="ascii"
    )
    result = audit(tmp_path, mode="working-tree")
    assert result.forbidden
    assert any("private absolute path" in finding for finding in result.findings)


@pytest.mark.parametrize(
    "private_root",
    [
        "C:" + "\\secret",
        "E:" + "\\datasets",
        "/" + "opt/private",
        "/" + "var/lib/project",
    ],
)
def test_public_audit_rejects_one_component_absolute_roots(
    tmp_path: Path, private_root: str,
) -> None:
    (tmp_path / "README.md").write_text(
        f"private root: {private_root}\n", encoding="ascii"
    )
    result = audit(tmp_path, mode="working-tree")
    assert result.forbidden
    assert any("private absolute path" in finding for finding in result.findings)


def test_staged_mode_reads_index_blobs_and_rejects_delete_rename(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("safe\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"],
        cwd=tmp_path,
        check=True,
    )
    safe = audit(tmp_path, mode="staged")
    assert not safe.forbidden, safe.findings

    (tmp_path / "README.md").write_text("https://x." + "try" + "cloudflare.com\n", encoding="utf-8")
    # The staged audit must use the index blob, not this unstaged content.
    still_safe = audit(tmp_path, mode="staged")
    assert not still_safe.forbidden, still_safe.findings

    (tmp_path / "README.md").write_text("safe\n", encoding="utf-8")
    subprocess.run(["git", "mv", "README.md", "RENAMED.md"], cwd=tmp_path, check=True)
    renamed = audit(tmp_path, mode="staged")
    assert renamed.forbidden
    assert any("rename" in finding.lower() for finding in renamed.findings)


def test_staged_mode_audits_unchanged_index_entries(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("safe\n", encoding="utf-8")
    historic = "/".join(("home", "admin_" + "mugen", "project", "checkpoint.pt"))
    (tmp_path / "history.md").write_text(
        f"historic path: {historic}\n", encoding="ascii"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"],
        cwd=tmp_path,
        check=True,
    )
    # Touch only a safe entry; the unchanged committed blob must still be audited.
    (tmp_path / "README.md").write_text("safe update\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    result = audit(tmp_path, mode="staged")
    assert result.forbidden
    assert any("private absolute path" in finding for finding in result.findings)


def test_staged_mode_rejects_detected_copy(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    content = "canonical tracked content\n"
    (tmp_path / "source.md").write_text(content, encoding="ascii")
    subprocess.run(["git", "add", "source.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "copy.md").write_text(content, encoding="ascii")
    subprocess.run(["git", "add", "copy.md"], cwd=tmp_path, check=True)
    result = audit(tmp_path, mode="staged")
    assert result.forbidden
    assert any("copy staged" in finding.lower() for finding in result.findings)


def test_staged_mode_rejects_delete(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("safe\n", encoding="ascii")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "rm", "-q", "README.md"], cwd=tmp_path, check=True)
    result = audit(tmp_path, mode="staged")
    assert result.forbidden
    assert any("delete staged" in finding.lower() for finding in result.findings)


def test_staged_mode_rejects_index_symlink_without_following_it(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    target = tmp_path / "secret.txt"
    target.write_text("private\n", encoding="utf-8")
    link = tmp_path / "public.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        # Windows environments without symlink privileges still exercise the other gates.
        return
    subprocess.run(["git", "add", "public.txt"], cwd=tmp_path, check=True)
    result = audit(tmp_path, mode="staged")
    assert result.forbidden
    assert any("symlink" in finding.lower() for finding in result.findings)


def test_cli_returns_nonzero_for_forbidden_surface(tmp_path: Path) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "x.txt").write_text("x", encoding="ascii")
    assert main(["--repo-root", str(tmp_path), "--mode", "working-tree"]) != 0
