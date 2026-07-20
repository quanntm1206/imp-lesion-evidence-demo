# Evidence-First Paper and Demo Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clean private-GitHub project, an evidence-bound paper PDF, and an exact fail-closed Loop206 live comparison website within 35 hours.

**Architecture:** A curated Git repository tracks only portable source, tests, compact evidence, paper source, and demo assets. A normalized evidence registry is generated from immutable Loop191/192/206 reports. The live candidate path reconstructs the exact Loop205 regional-saliency and Loop206 active-contour deployment prior from the leakage-safe train partition, proves parity against the frozen Loop206 cache, then supplies the fourth input channel to the real control/candidate checkpoints. Gradio reads only validated registries and model services; paper tables read the same evidence registry.

**Tech Stack:** Python 3.12, uv, PyTorch, segmentation-models-pytorch 0.5.0, OpenCV, scikit-image, scikit-learn, Gradio, pytest, MiKTeX/latexmk, GitHub Actions, Cloudflare Tunnel.

## Global Constraints

- Delivery time-box: 35 hours.
- Main workstation: RTX 5060 Ti 16 GB; model inference, artifact generation, GPU verification, demo serving.
- Laptop: RTX 4060 8 GB, initially empty; clean clone, paper build, citation review, CPU tests, UI smoke tests.
- Main paper comparison: Loop191 IMP-SegFormer-B3 versus Loop192 nnU-Net v2 on Clean-v3 validation.
- Controlled ablation: Loop206 three-seed zero-channel control versus contour-channel candidate on train-screen only.
- Loop170 is always labeled `legacy_patient_contaminated`; never rank it with Clean-v3.
- No SOTA, protected-test, statistical-superiority, clinical, or diagnostic claim.
- No uploaded ground truth means no Dice, IoU, BF1, HD95, or ASSD output.
- Candidate inference must never substitute a zero, hand-drawn, control-model, or approximate contour channel.
- Arbitrary upload is enabled only after byte-identical parity against the frozen Loop206 holdout contour cache.
- Dataset, `.artifacts`, checkpoints, logs, secrets, tunnel credentials, generated predictions, and environments remain outside Git.
- GitHub remote is private during rescue. Model weights remain local unless separate license review approves transfer.
- Every task uses TDD, focused verification, and an independent commit.

---

## File Map

### Repository and environment

- Create `.gitignore`: allowlist the portable rescue project; exclude local legacy/artifact trees.
- Modify `pyproject.toml`: add demo dependencies and `lesion-demo`/evidence-builder entry points.
- Modify `README.md`: replace WSL-only instructions with Windows main-machine and laptop workflows.
- Create `.github/workflows/ci.yml`: clean-clone CPU verification.
- Create `scripts/bootstrap_windows.ps1`: deterministic Windows setup.
- Create `docs/runbooks/two-machine-delivery.md`: branch ownership and artifact transfer policy.

### Evidence

- Create `src/lesion_robustness/evidence_registry.py`: schema, validation, hashing, source extraction.
- Create `scripts/demo/build_evidence_registry.py`: CLI builder from local immutable reports.
- Create `demo/data/evidence_registry.json`: generated compact registry used by paper and UI.
- Create `tests/demo/test_evidence_registry.py`: classification, number, hash, and claim-policy tests.

### Exact Loop206 deployment prior

- Create `src/lesion_robustness/demo/__init__.py`: demo package marker.
- Create `src/lesion_robustness/demo/data_index.py`: portable manifest-to-local-dataset resolver.
- Create `src/lesion_robustness/demo/loop206_prior.py`: fit, serialize, load, and apply the Loop205/206 prior.
- Create `scripts/demo/build_loop206_prior.py`: main-workstation artifact builder and parity gate.
- Create `configs/demo/loop206_live.yaml`: self-contained inference contract without local paths.
- Create `demo/model_registry.example.json`: environment-variable model paths and frozen hashes.
- Create `tests/demo/test_data_index.py`: source identity and leakage-policy tests.
- Create `tests/demo/test_loop206_prior.py`: deterministic fitting, serialization, and parity tests.

### Model inference and metrics

- Create `src/lesion_robustness/demo/model_service.py`: validated dual-model loading and sequential inference.
- Create `src/lesion_robustness/demo/geometry.py`: resize/restore contract and overlays.
- Create `src/lesion_robustness/demo/metrics_service.py`: optional-GT validation and metrics.
- Create `tests/demo/test_model_service.py`: checkpoint and fourth-channel fail-closed tests.
- Create `tests/demo/test_geometry.py`: non-square input and restoration tests.
- Create `tests/demo/test_metrics_service.py`: perfect, empty, mismatch, and no-GT tests.

### Website and deployment

- Create `src/lesion_robustness/demo/app.py`: Gradio app factory and CLI.
- Create `src/lesion_robustness/demo/presentation.py`: tables, receipts, disclaimers, rendering helpers.
- Create `src/lesion_robustness/demo/theme.css`: deliberate dermoscopy-lab visual language.
- Create `scripts/demo/run_demo.ps1`: local launch.
- Create `scripts/demo/run_tunnel.ps1`: tunnel launch without embedded credentials.
- Create `tests/demo/test_app.py`: UI construction and callback contract.
- Create `tests/demo/test_presentation.py`: evidence labels and legacy warning tests.
- Create `docs/runbooks/demo-operations.md`: startup, health check, cleanup, fallback.

### Paper

- Create `paper/clean_v3_loop206/main.tex`: manuscript root.
- Create `paper/clean_v3_loop206/sections/01_introduction.tex` through `10_conclusion.tex`: complete paper.
- Create `paper/clean_v3_loop206/references.bib`: verified primary references only.
- Create `paper/clean_v3_loop206/tables/*.tex`: generated evidence-bound tables.
- Create `paper/clean_v3_loop206/figures/*`: editable source plus PDF/PNG exports.
- Create `paper/clean_v3_loop206/artifact_manifest.json`: paper input hashes.
- Create `scripts/paper/build_clean_v3_tables.py`: deterministic table generator.
- Create `scripts/paper/audit_clean_v3_paper.py`: unfinished-marker, claim, evidence, reference audit.
- Create `tests/demo/test_paper_tables.py`: exact number/evidence-class tests.
- Create `tests/demo/test_paper_audit.py`: forbidden-claim and missing-source tests.

---

### Task 1: Curate the Git Repository and Dependency Contract

**Files:**
- Create: `.gitignore`
- Modify: `pyproject.toml`
- Test: `tests/demo/test_repository_contract.py`

**Interfaces:**
- Consumes: existing Python package under `src/lesion_robustness`.
- Produces: extras `demo`, commands `lesion-demo` and `lesion-build-evidence`, portable tracked-file policy.

- [ ] **Step 1: Write the repository-contract test**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_demo_dependency_and_entry_points_are_declared() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(config["project"]["optional-dependencies"]["demo"]) >= {
        "gradio>=5,<7",
        "joblib>=1.3",
    }
    scripts = config["project"]["scripts"]
    assert scripts["lesion-demo"] == "lesion_robustness.demo.app:main"
    assert scripts["lesion-build-evidence"] == "lesion_robustness.evidence_registry:main"


def test_gitignore_blocks_private_runtime_assets() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="ascii")
    for pattern in (".artifacts/", "runs/", "data/", "*.pt", "*.pth", ".env"):
        assert pattern in text
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `uv run --extra dev pytest tests/demo/test_repository_contract.py -v`

Expected: FAIL because `.gitignore`, the `demo` extra, and both entry points do not exist.

- [ ] **Step 3: Create the allowlist-oriented `.gitignore`**

```gitignore
# Runtime and research evidence
.artifacts/
runs/
checkpoints/
data/
demo_runtime/
logs/

# Model/data formats
*.pt
*.pth
*.ckpt
*.onnx
*.mmap
*.npy
*.npz
*.xlsx

# Python
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/

# Secrets and uploads
.env
.env.*
!.env.example
uploads/
cloudflared*.log

# LaTeX build products
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
```

- [ ] **Step 4: Add the demo extra and entry points to `pyproject.toml`**

```toml
demo = [
  "gradio>=5,<7",
  "joblib>=1.3",
]

[project.scripts]
lesion-demo = "lesion_robustness.demo.app:main"
lesion-build-evidence = "lesion_robustness.evidence_registry:main"
```

Preserve every existing dependency, extra, and script entry.

- [ ] **Step 5: Lock dependencies and pass the test**

Run: `uv lock --upgrade-package gradio --upgrade-package joblib`

Run: `uv run --extra dev pytest tests/demo/test_repository_contract.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore pyproject.toml uv.lock tests/demo/test_repository_contract.py
git commit -m "build: define portable demo environment"
```

### Task 2: Build the Evidence Registry

**Files:**
- Create: `src/lesion_robustness/evidence_registry.py`
- Create: `scripts/demo/build_evidence_registry.py`
- Create: `demo/data/evidence_registry.json`
- Test: `tests/demo/test_evidence_registry.py`

**Interfaces:**
- Consumes: Loop191, Loop192, Loop206 final JSON reports; optional Loop170 legacy table source.
- Produces: `build_registry(sources: EvidenceSources) -> dict[str, object]`, `validate_registry(payload) -> None`, canonical ASCII JSON.

- [ ] **Step 1: Write schema and classification tests**

```python
from lesion_robustness.evidence_registry import EvidenceSources, build_registry, validate_registry


def test_registry_separates_scientific_validation_screen_and_legacy(frozen_reports) -> None:
    registry = build_registry(EvidenceSources(**frozen_reports))
    validate_registry(registry)
    rows = {row["model_id"]: row for row in registry["observations"]}
    assert rows["L191-C0-clean-v3-IMP-control"]["evidence_class"] == "protected_validation"
    assert rows["L192-nnUNet-v2-raw-100ep"]["robust_dice"] == 0.9019177076063616
    assert rows["L206-contour-vs-control"]["evidence_class"] == "train_screen"
    assert rows["Loop170-IMP"]["evidence_class"] == "legacy_patient_contaminated"
    assert registry["scientific_sota_status"] == "not_established"


def test_registry_rejects_legacy_as_comparable(frozen_reports) -> None:
    registry = build_registry(EvidenceSources(**frozen_reports))
    legacy = next(row for row in registry["observations"] if row["model_id"] == "Loop170-IMP")
    legacy["scientific_comparable"] = True
    try:
        validate_registry(registry)
    except ValueError as exc:
        assert "legacy" in str(exc).lower()
    else:
        raise AssertionError("legacy evidence must fail closed")
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run --extra dev pytest tests/demo/test_evidence_registry.py -v`

Expected: FAIL with `ModuleNotFoundError: lesion_robustness.evidence_registry`.

- [ ] **Step 3: Implement immutable source types and canonical serialization**

```python
@dataclass(frozen=True)
class EvidenceSources:
    loop191: Path
    loop192: Path
    loop206: Path
    loop170_locked_panel: Path
    loop170_bootstrap: Path


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

`build_registry` must emit exact source paths relative to the project root, source SHA-256, dataset version, partition, evidence class, metric contract, seed count, metrics, confidence intervals, `scientific_comparable`, and limitations. It must parse values rather than duplicate constants in the builder.

- [ ] **Step 4: Implement validation rules**

Reject unknown evidence classes, missing hashes, non-finite values, duplicate model IDs, protected-test claims, legacy scientific comparability, Loop206 validation/test classification, and Loop191/192 multi-seed claims.

- [ ] **Step 5: Generate and verify the tracked registry**

Run:

```powershell
uv run --extra analysis python scripts/demo/build_evidence_registry.py `
  --loop191 .artifacts/preprocessing_search/current_bdou_loop191_raw_rater_uncertainty_report.json `
  --loop192 .artifacts/preprocessing_search/current_bdou_loop192_nnunet_clean_v3_report.json `
  --loop206 .artifacts/preprocessing_search/current_bdou_loop206_final_closure_report.json `
  --loop170-panel reports/paper_loop170_overleaf/tables/locked_panel_results.tex `
  --loop170-bootstrap reports/paper_loop170_overleaf/tables/bootstrap_deltas.tex `
  --output demo/data/evidence_registry.json
```

Expected: `registry_status=valid`, `scientific_sota_status=not_established`, five source hashes printed.

- [ ] **Step 6: Pass tests and commit**

Run: `uv run --extra dev pytest tests/demo/test_evidence_registry.py -v`

```powershell
git add src/lesion_robustness/evidence_registry.py scripts/demo/build_evidence_registry.py demo/data/evidence_registry.json tests/demo/test_evidence_registry.py
git commit -m "feat: add protocol-bound evidence registry"
```

### Task 3: Resolve the Loop206 Dataset Without Reintroducing Absolute Paths

**Files:**
- Create: `src/lesion_robustness/demo/data_index.py`
- Create: `scripts/demo/index_loop206_dataset.py`
- Test: `tests/demo/test_data_index.py`

**Interfaces:**
- Consumes: `data/splits/loop206_pilot_manifest.csv`, one or more local dataset roots.
- Produces: `resolve_loop206_rows(manifest: Path, roots: Sequence[Path]) -> list[ResolvedRow]` and untracked `demo_runtime/loop206_dataset_index.json`.

- [ ] **Step 1: Write resolver tests**

```python
def test_resolver_matches_stem_and_both_hashes(tmp_path: Path) -> None:
    image, mask, manifest = make_manifest_case(tmp_path, sample_id="ISIC_0000001")
    rows = resolve_loop206_rows(manifest, [tmp_path / "dataset"])
    assert len(rows) == 1
    assert rows[0].image_path == image.resolve()
    assert rows[0].mask_path == mask.resolve()


def test_resolver_rejects_missing_duplicate_or_hash_mismatch(tmp_path: Path) -> None:
    manifest = make_ambiguous_manifest_case(tmp_path)
    with pytest.raises(ValueError, match="unique hash-verified image"):
        resolve_loop206_rows(manifest, [tmp_path])
```

- [ ] **Step 2: Verify failure**

Run: `uv run --extra dev pytest tests/demo/test_data_index.py -v`

Expected: FAIL because `lesion_robustness.demo.data_index` does not exist.

- [ ] **Step 3: Implement the indexer**

Use `data_manifest.sha256_file` and `data_manifest.sha256_rgb`. Index images by basename; index masks by the original mask basename. Require exactly 384 rows, `308 fit + 76 holdout`, source split `train`, unique `loop205_group_key`, hashes matching `sha256_raw` and `sha256_rgb`, and zero fit/holdout group overlap. Serialize only relative paths under user-provided roots; never serialize historic `/home/admin_mugen/...` paths.

- [ ] **Step 4: Run against available roots**

Run:

```powershell
uv run --extra analysis python scripts/demo/index_loop206_dataset.py `
  --manifest data/splits/loop206_pilot_manifest.csv `
  --root E:\datasets `
  --output demo_runtime/loop206_dataset_index.json
```

Expected when data is absent: nonzero exit with exact missing counts per `isic2016`, `isic2017`, and `isic2018`; no partial index.

Expected after dataset restoration: `rows=384 fit=308 holdout=76 hash_mismatches=0 overlap=0`.

- [ ] **Step 5: Pass tests and commit**

Run: `uv run --extra dev pytest tests/demo/test_data_index.py -v`

```powershell
git add src/lesion_robustness/demo/__init__.py src/lesion_robustness/demo/data_index.py scripts/demo/index_loop206_dataset.py tests/demo/test_data_index.py
git commit -m "feat: resolve portable Loop206 data"
```

### Task 4: Build and Prove the Exact Loop206 Deployment Prior

**Files:**
- Create: `src/lesion_robustness/demo/loop206_prior.py`
- Create: `scripts/demo/build_loop206_prior.py`
- Create: `configs/demo/loop206_live.yaml`
- Test: `tests/demo/test_loop206_prior.py`

**Interfaces:**
- Consumes: 308 resolved fit rows, frozen Loop205 config, `neutral_mid_30_s2`, candidate holdout cache.
- Produces: `Loop206Prior.predict(image_rgb_u8) -> np.ndarray`, untracked `demo_runtime/loop206_prior.joblib`, immutable receipt JSON.

- [ ] **Step 1: Write deterministic unit tests**

```python
def test_prior_round_trip_is_deterministic(tiny_fit_rows, tmp_path: Path) -> None:
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1)
    path = tmp_path / "prior.joblib"
    save_prior(prior, path)
    loaded = load_prior(path, expected_sha256=sha256_file(path))
    first = loaded.predict(tiny_fit_rows[0].image)
    second = loaded.predict(tiny_fit_rows[0].image.copy())
    np.testing.assert_array_equal(first, second)


def test_candidate_channel_requires_parity_receipt(tmp_path: Path) -> None:
    prior = make_prior(parity_passed=False)
    with pytest.raises(RuntimeError, match="parity"):
        prior.predict(np.zeros((384, 384, 3), dtype=np.uint8))
```

- [ ] **Step 2: Verify failure**

Run: `uv run --extra dev --extra analysis pytest tests/demo/test_loop206_prior.py -v`

Expected: FAIL because `loop206_prior` does not exist.

- [ ] **Step 3: Implement the deployment-fit contract**

Reuse `loop205_protocol.extract_region_features`, `compute_region_targets`, `fit_region_forest`, `predict_saliency_map`, and `select_train_only_threshold`. Use the exact Loop206 runtime preprocessing and all three preregistered views for the 308 fit rows. Use out-of-bag predictions for threshold calibration. Apply `loop206_active_contour.refine_active_contour` with canonical `neutral_mid_30_s2`.

The serialized artifact contains the regressor, selected threshold, Loop205/206 config payloads, manifest hash, fit-group hash, feature names, sklearn version, and code hashes. Loading rejects any mismatch.

- [ ] **Step 4: Implement holdout byte-parity gating**

For all 76 holdout clean rows, regenerate the preprocessed RGB, probability, and active contour. Compare each contour byte-for-byte with `.artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_candidate/contours.uint8.mmap`. Require `76/76` exact matches and input RGB hashes matching the candidate manifest. Any mismatch deletes the temporary artifact and writes a failed receipt.

- [ ] **Step 5: Add the portable inference config**

```yaml
schema_version: loop206.demo.live.v1
image_size: [384, 384]
model:
  name: segformer_mit
  in_channels: 3
  input_channels: 4
  encoder_in_channels: 3
  out_channels: 1
  base_channels: 32
  smp_encoder: mit_b3
  encoder_weights: null
  edge_aux: false
  input_adapter:
    enabled: true
    initialization: rgb_identity_extra_zero
preprocessing:
  contrast_stretch:
    enabled: true
    lower_percentile: 1.0
    upper_percentile: 99.0
  clahe:
    enabled: true
    color_space: lab
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  filter:
    type: median
    kernel_size: 3
  color_constancy: {enabled: false}
postprocessing:
  threshold: 0.5
  close_kernel: 5
  fill_holes: true
  min_area_ratio: 0.005
  keep_largest: true
metric:
  boundary_tolerance: 2
  empty_boundary_distance_policy: image_diagonal
```

- [ ] **Step 6: Build the real artifact**

Run:

```powershell
uv run --extra analysis python scripts/demo/build_loop206_prior.py `
  --dataset-index demo_runtime/loop206_dataset_index.json `
  --candidate-cache .artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_candidate/manifest.json `
  --output demo_runtime/loop206_prior.joblib `
  --receipt demo_runtime/loop206_prior_receipt.json `
  --n-jobs 4
```

Expected: `fit_groups=308 holdout_parity=76/76 parity_passed=true`.

- [ ] **Step 7: Pass tests and commit**

Run: `uv run --extra dev --extra analysis pytest tests/demo/test_loop206_prior.py -v`

```powershell
git add src/lesion_robustness/demo/loop206_prior.py scripts/demo/build_loop206_prior.py configs/demo/loop206_live.yaml tests/demo/test_loop206_prior.py
git commit -m "feat: build exact Loop206 deployment prior"
```

### Task 5: Implement Dual-Model Inference and Geometry

**Files:**
- Create: `src/lesion_robustness/demo/model_service.py`
- Create: `src/lesion_robustness/demo/geometry.py`
- Create: `demo/model_registry.example.json`
- Test: `tests/demo/test_model_service.py`
- Test: `tests/demo/test_geometry.py`

**Interfaces:**
- Produces: `Loop206ComparisonService.compare(image: np.ndarray) -> ComparisonResult`.
- `ComparisonResult` includes original RGB, control/candidate probabilities, restored masks, overlays, latency, device, model IDs, checkpoint hashes, and prior receipt hash.

- [ ] **Step 1: Write fail-closed service tests**

```python
def test_service_builds_zero_and_exact_candidate_channels(fake_models, parity_prior) -> None:
    service = Loop206ComparisonService(fake_models.control, fake_models.candidate, parity_prior)
    result = service.compare(np.full((240, 320, 3), 128, dtype=np.uint8))
    np.testing.assert_array_equal(fake_models.control.last_input[0, 3], 0.0)
    assert fake_models.candidate.last_input[0, 3].max() == 1.0
    assert result.control_mask.shape == (240, 320)
    assert result.candidate_mask.shape == (240, 320)


def test_service_rejects_hash_mismatch(tmp_path: Path) -> None:
    registry = make_model_registry(tmp_path, declared_hash="0" * 64)
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_model_registry(registry)
```

- [ ] **Step 2: Verify failure**

Run: `uv run --extra dev pytest tests/demo/test_model_service.py tests/demo/test_geometry.py -v`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement geometry and presentation arrays**

`prepare_image` validates RGB uint8, maximum 16 megapixels, minimum side 32, and resizes to `384x384`. `restore_probability` bilinearly restores probabilities to original geometry before thresholding. `overlay_mask` renders a bounded alpha overlay without changing the source array.

- [ ] **Step 4: Implement model loading and sequential inference**

Build the four-channel SMP model with `encoder_weights=None`, load `state["model"]` strictly, set eval mode, and validate checkpoint hashes. Load one seed-matched pair, seed 206 by default. Run control then candidate under `torch.inference_mode()` and a process lock. Clear only temporary tensors; keep both models resident.

- [ ] **Step 5: Add the registry template**

```json
{
  "schema_version": "loop206.demo.models.v1",
  "control": {
    "model_id": "L206-control-s206",
    "checkpoint_env": "IMP_LOOP206_CONTROL_CHECKPOINT",
    "checkpoint_sha256": "be606b0a0940839b019ea60117dda4b27f9b8f04d54306b5b676f2c29516fcef"
  },
  "candidate": {
    "model_id": "L206-contour-channel-s206",
    "checkpoint_env": "IMP_LOOP206_CANDIDATE_CHECKPOINT",
    "checkpoint_sha256": "afb86b2a5161189369dbc3c985e78f214c305470661048c6643726612f57638b"
  },
  "prior_env": "IMP_LOOP206_PRIOR",
  "prior_receipt_env": "IMP_LOOP206_PRIOR_RECEIPT"
}
```

Before committing, derive both checkpoint hashes from the actual selected files and replace the example values if they differ. Tests require exact equality.

- [ ] **Step 6: Run CPU tests and real GPU smoke test**

Run: `uv run --extra dev pytest tests/demo/test_model_service.py tests/demo/test_geometry.py -v`

Run:

```powershell
$env:IMP_LOOP206_CONTROL_CHECKPOINT='runs/loop206-control-train-screen-pilot20-checkpoints/best.pt'
$env:IMP_LOOP206_CANDIDATE_CHECKPOINT='runs/loop206-contour-channel-train-screen-pilot20-checkpoints/best.pt'
$env:IMP_LOOP206_PRIOR='demo_runtime/loop206_prior.joblib'
$env:IMP_LOOP206_PRIOR_RECEIPT='demo_runtime/loop206_prior_receipt.json'
uv run --extra train --extra analysis python -m lesion_robustness.demo.model_service --smoke-image demo_runtime/smoke.jpg
```

Expected: both models load strictly; two `384x384` predictions; original-size masks; finite latency; no OOM.

- [ ] **Step 7: Commit**

```powershell
git add src/lesion_robustness/demo/model_service.py src/lesion_robustness/demo/geometry.py demo/model_registry.example.json tests/demo/test_model_service.py tests/demo/test_geometry.py
git commit -m "feat: compare Loop206 models exactly"
```

### Task 6: Implement Optional Ground-Truth Metrics

**Files:**
- Create: `src/lesion_robustness/demo/metrics_service.py`
- Test: `tests/demo/test_metrics_service.py`

**Interfaces:**
- Produces: `evaluate_optional_ground_truth(control, candidate, gt | None) -> dict[str, dict[str, float]] | None`.

- [ ] **Step 1: Write metric tests**

```python
def test_no_ground_truth_returns_none() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    assert evaluate_optional_ground_truth(mask, mask, None) is None


def test_perfect_masks_return_unit_overlap_and_zero_distance() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    metrics = evaluate_optional_ground_truth(mask, mask, mask)
    for arm in ("control", "candidate"):
        assert metrics[arm]["dice"] == 1.0
        assert metrics[arm]["iou"] == 1.0
        assert metrics[arm]["boundary_f1"] == 1.0
        assert metrics[arm]["hd95"] == 0.0
        assert metrics[arm]["assd"] == 0.0
```

- [ ] **Step 2: Implement strict mask validation**

Accept grayscale/RGB masks, reject alpha-only, non-finite, geometry mismatch, all-soft ambiguous masks, and masks larger than 16 megapixels. Threshold decoded mask at 127. Reuse `segmentation_metrics(..., include_boundary=True, boundary_tolerance=2, empty_boundary_distance_policy="image_diagonal")`.

- [ ] **Step 3: Pass tests and commit**

Run: `uv run --extra dev pytest tests/demo/test_metrics_service.py -v`

```powershell
git add src/lesion_robustness/demo/metrics_service.py tests/demo/test_metrics_service.py
git commit -m "feat: add optional ground-truth metrics"
```

### Task 7: Build the Gradio Website

**Files:**
- Create: `src/lesion_robustness/demo/app.py`
- Create: `src/lesion_robustness/demo/presentation.py`
- Create: `src/lesion_robustness/demo/theme.css`
- Test: `tests/demo/test_app.py`
- Test: `tests/demo/test_presentation.py`

**Interfaces:**
- Consumes: `Loop206ComparisonService`, evidence registry, optional ground truth.
- Produces: `create_app(service, registry) -> gr.Blocks`, CLI host/port/share options.

- [ ] **Step 1: Invoke UI skills before visual code**

Use `$design-taste` to establish a dermoscopy research-console language: warm bone background, oxidized teal, vermilion alerts, condensed display type, clinical-grid texture, strong evidence badges, no purple gradient, responsive two-column comparison collapsing to one column.

- [ ] **Step 2: Write app contract tests**

```python
def test_prediction_without_gt_has_no_accuracy_payload(fake_service, registry) -> None:
    response = run_comparison(fake_service, registry, sample_image(), None)
    assert response.metrics_markdown == "Ground truth not supplied; accuracy metrics are unavailable."


def test_legacy_rows_always_render_warning(registry) -> None:
    html = render_legacy_table(registry)
    assert "legacy_patient_contaminated" in html
    assert "13" in html and "3" in html
```

- [ ] **Step 3: Implement the three-view app**

Tabs:

1. `Live Compare`: image upload, optional GT, run button, original/overlays/masks, metric table, latency, hashes, downloadable JSON receipt.
2. `Clean-v3 Evidence`: Loop191/192 validation table and Loop206 negative-ablation CI.
3. `Legacy Audit`: Loop170 table behind a persistent patient-contamination banner.

The callback returns an error card rather than partial outputs when either arm fails. Queue concurrency is exactly one.

- [ ] **Step 4: Implement CSS and mobile behavior**

Use CSS variables `--paper`, `--ink`, `--teal`, `--rust`, `--line`, and `--warning`. Set a non-default heading family with local/web-safe fallback, stagger the initial evidence cards once, and disable motion under `prefers-reduced-motion`. At widths below 760 px, stack controls, comparison panels, and evidence cards.

- [ ] **Step 5: Pass tests and render locally**

Run: `uv run --extra dev --extra demo pytest tests/demo/test_app.py tests/demo/test_presentation.py -v`

Run: `uv run --extra train --extra analysis --extra demo lesion-demo --host 127.0.0.1 --port 7860`

Expected: app starts; both model hashes shown; no metric table without GT; legacy warning visible.

- [ ] **Step 6: Invoke `$design-review`**

Open the local app in the in-app browser, capture desktop and mobile screenshots, inspect typography, hierarchy, overflow, evidence labels, error state, and real output state. Correct every P0/P1 visual or accessibility issue before commit.

- [ ] **Step 7: Commit**

```powershell
git add src/lesion_robustness/demo/app.py src/lesion_robustness/demo/presentation.py src/lesion_robustness/demo/theme.css tests/demo/test_app.py tests/demo/test_presentation.py
git commit -m "feat: add evidence-first comparison website"
```

### Task 8: Generate Paper Tables from the Same Registry

**Files:**
- Create: `scripts/paper/build_clean_v3_tables.py`
- Create: `paper/clean_v3_loop206/tables/evidence_scope.tex`
- Create: `paper/clean_v3_loop206/tables/clean_v3_validation.tex`
- Create: `paper/clean_v3_loop206/tables/loop206_ablation.tex`
- Create: `paper/clean_v3_loop206/tables/legacy_loop170.tex`
- Test: `tests/demo/test_paper_tables.py`

**Interfaces:**
- Consumes: `demo/data/evidence_registry.json`.
- Produces: deterministic LaTeX tables and `paper/clean_v3_loop206/artifact_manifest.json`.

- [ ] **Step 1: Write exact-value tests**

```python
def test_clean_v3_table_contains_scoped_point_estimates(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["clean_v3_validation"].read_text(encoding="utf-8")
    assert "0.8959" in text
    assert "0.9019" in text
    assert "validation" in text.lower()
    assert "SOTA" not in text


def test_legacy_table_carries_contamination_label(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["legacy_loop170"].read_text(encoding="utf-8")
    assert "legacy\\_patient\\_contaminated" in text
```

- [ ] **Step 2: Implement deterministic formatting**

Round displayed values to four decimals while preserving full precision in the artifact manifest. Escape LaTeX text. Emit captions and labels with explicit dataset, partition, metric contract, and seed count. Never add bold-best markup across evidence classes.

- [ ] **Step 3: Generate, test, and commit**

Run: `uv run --extra analysis python scripts/paper/build_clean_v3_tables.py --registry demo/data/evidence_registry.json --paper-dir paper/clean_v3_loop206`

Run: `uv run --extra dev pytest tests/demo/test_paper_tables.py -v`

```powershell
git add scripts/paper/build_clean_v3_tables.py paper/clean_v3_loop206/tables paper/clean_v3_loop206/artifact_manifest.json tests/demo/test_paper_tables.py
git commit -m "feat: generate evidence-bound paper tables"
```

### Task 9: Draft the Complete Paper

**Files:**
- Create: `paper/clean_v3_loop206/main.tex`
- Create: `paper/clean_v3_loop206/sections/01_introduction.tex`
- Create: `paper/clean_v3_loop206/sections/02_related_work.tex`
- Create: `paper/clean_v3_loop206/sections/03_data_protocol.tex`
- Create: `paper/clean_v3_loop206/sections/04_methods.tex`
- Create: `paper/clean_v3_loop206/sections/05_experiments.tex`
- Create: `paper/clean_v3_loop206/sections/06_results.tex`
- Create: `paper/clean_v3_loop206/sections/07_discussion.tex`
- Create: `paper/clean_v3_loop206/sections/08_limitations_ethics.tex`
- Create: `paper/clean_v3_loop206/sections/09_reproducibility.tex`
- Create: `paper/clean_v3_loop206/sections/10_conclusion.tex`
- Create: `paper/clean_v3_loop206/references.bib`

**Interfaces:**
- Consumes: generated tables, verified local artifacts, primary literature.
- Produces: complete article-style manuscript with no venue-rank claim.

- [ ] **Step 1: Invoke `$research`, `$ml-cpv-research`, and `$ml-paper-writer`**

Verify primary sources for SegFormer, nnU-Net, U-Net, EGE-UNet, DermoSegDiff, ISIC challenge data, segmentation metrics, and leakage-aware ML evaluation. Keep arXiv-only status explicit. Do not claim Q1/Q2 or venue rank.

- [ ] **Step 2: Write title, abstract, and introduction**

Use the approved title. Abstract must contain: Clean-v3 validation scope; Loop191 Dice `0.8959`; Loop192 Dice `0.9019`; Loop206 Dice delta `-0.0313` with CI `[-0.0491,-0.0156]`; missing protected test; no superiority claim.

Introduction contributions:

1. leakage-audited evidence hierarchy;
2. bounded SegFormer/nnU-Net Clean-v3 validation comparison;
3. three-seed negative contour-channel ablation;
4. reproducible evidence/demo artifact with explicit limitations.

- [ ] **Step 3: Write related work and protocol**

Organize related work by claim space, not chronology. Protocol must state 2,869 Clean-v3 rows, `2008/431/430` train/validation/test counts, group-disjoint audit, corruptions, partition access, Loop192 `256→384` old-geometry limitation, and Loop170 contamination.

- [ ] **Step 4: Write methods and experiments**

Describe the exact SegFormer-B3/SMP U-Net decoder implementation used by the repo, raw RGB nnU-Net v2, Loop205 regional forest, Loop206 active contour, zero-channel control, input adapter, preprocessing, postprocessing, three paired seeds, cluster bootstrap, hardware, and resource limits. Separate architecture comparison from causal ablation.

- [ ] **Step 5: Write results, discussion, limitations, reproducibility, conclusion**

Results quote only generated tables. Discussion explains metric trade-offs and failure mechanisms without causally attributing Loop191/192 differences. Limitations name single-run validation, no protected test, metric geometry, absent complete local dataset at audit start, and non-clinical deployment. Conclusion states `scientific_sota_status = not_established`.

- [ ] **Step 6: Compile and inspect**

Run:

```powershell
Push-Location paper/clean_v3_loop206
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

Expected: exit 0; no undefined references/citations; `main.pdf` produced.

- [ ] **Step 7: Commit**

```powershell
git add paper/clean_v3_loop206/main.tex paper/clean_v3_loop206/sections paper/clean_v3_loop206/references.bib
git commit -m "docs: draft leakage-aware comparison paper"
```

### Task 10: Create Publication Figures and Qualitative Demo Evidence

**Files:**
- Create: `paper/clean_v3_loop206/figures/evidence_pipeline.drawio`
- Create: `paper/clean_v3_loop206/figures/evidence_pipeline.pdf`
- Create: `paper/clean_v3_loop206/figures/loop206_delta.pdf`
- Create: `paper/clean_v3_loop206/figures/qualitative_demo.pdf`
- Modify: relevant paper sections and artifact manifest.

**Interfaces:**
- Consumes: registry, real demo receipts, selected user-independent examples.
- Produces: claim-bound figures with editable sources.

- [ ] **Step 1: Invoke `$drawio-skill` with `style_preset="ml-journal"`**

Build one pipeline figure showing evidence classes, model comparison, Loop206 ablation, demo inference, and claim gates. Preserve `.drawio`; export vector PDF.

- [ ] **Step 2: Generate the Loop206 delta plot**

Plot Dice and BF1 point deltas with 95% CI, zero reference line, and `train-screen` label. Do not mix Loop191/192 point estimates into this confidence-interval plot.

- [ ] **Step 3: Generate qualitative comparisons from real demo receipts**

Use three non-protected images whose provenance permits local display. Show original, control mask, candidate mask, disagreement overlay, and GT only when authorized. Caption every panel `illustrative; not protected-test evidence`.

- [ ] **Step 4: Register figure hashes and compile**

Update `artifact_manifest.json` with path, SHA-256, caption, label, supported claim, evidence class, and generation command. Recompile with `latexmk`.

- [ ] **Step 5: Commit**

```powershell
git add paper/clean_v3_loop206/figures paper/clean_v3_loop206/sections paper/clean_v3_loop206/artifact_manifest.json
git commit -m "docs: add evidence-bound paper figures"
```

### Task 11: Add Paper and Demo Audits

**Files:**
- Create: `scripts/paper/audit_clean_v3_paper.py`
- Create: `tests/demo/test_paper_audit.py`
- Create: `docs/runbooks/demo-operations.md`

**Interfaces:**
- Produces: machine-readable audit receipt and nonzero exit on unsupported output.

- [ ] **Step 1: Write failing audit tests**

```python
@pytest.mark.parametrize("forbidden", ["state-of-the-art", "statistically superior", "clinical-grade"])
def test_audit_rejects_forbidden_claims(tmp_path: Path, forbidden: str) -> None:
    paper = make_minimal_paper(tmp_path, body=forbidden)
    result = audit_paper(paper, REGISTRY)
    assert not result.passed


def test_audit_requires_all_registry_sources(tmp_path: Path) -> None:
    paper = make_minimal_paper(tmp_path, body="No numeric evidence")
    result = audit_paper(paper, REGISTRY)
    assert "missing evidence mapping" in result.errors
```

- [ ] **Step 2: Implement the audit**

Scan all TeX, BibTeX, tables, captions, README, and demo registry. Reject unfinished markers, undefined citation keys, unsupported numeric results, protected-test/SOTA language, unlabeled Loop170 values, hidden no-GT metrics, missing figure hashes, and source-hash drift.

- [ ] **Step 3: Write the operations runbook**

Document environment variables, model/prior hash verification, one-worker queue, local health check, upload cleanup, tunnel start/stop, failure recovery, and local-only fallback. Include explicit non-clinical warning.

- [ ] **Step 4: Run audits and commit**

Run: `uv run --extra dev --extra analysis python scripts/paper/audit_clean_v3_paper.py --paper paper/clean_v3_loop206 --registry demo/data/evidence_registry.json --receipt demo_runtime/paper_audit.json`

Expected: `passed=true errors=0`.

```powershell
git add scripts/paper/audit_clean_v3_paper.py tests/demo/test_paper_audit.py docs/runbooks/demo-operations.md
git commit -m "test: gate paper and demo claims"
```

### Task 12: Bootstrap the Laptop and Private GitHub Workflow

**Files:**
- Create: `scripts/bootstrap_windows.ps1`
- Create: `docs/runbooks/two-machine-delivery.md`
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

**Interfaces:**
- Produces: repeatable clean clone, private remote, CI, paper build, CPU checks.

- [ ] **Step 1: Write the bootstrap script**

```powershell
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required' }
uv sync --python 3.12 --extra dev --extra analysis --extra demo
uv run pytest tests/demo -q
Push-Location paper/clean_v3_loop206
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

- [ ] **Step 2: Add CPU CI**

CI checks out the repository, installs uv/Python 3.12, syncs `dev,analysis,demo`, runs `pytest tests/demo -q`, builds evidence tables from the tracked registry, audits paper claims, and compiles LaTeX when a TeX runner is available. GPU integration remains a required local receipt.

- [ ] **Step 3: Write branch and artifact policy**

Main workstation uses `main` for integration and `demo-runtime` for source changes. Laptop uses `paper-review`. Exchange code by push/pull. Exchange weights/prior only by private LAN/USB after hash verification; never through GitHub.

- [ ] **Step 4: Create the private GitHub remote**

Run:

```powershell
gh auth status
gh repo create imp-lesion-evidence-demo --private --source . --remote origin --push
```

If the repository already exists, resolve its authenticated SSH URL and add it; never create a public fallback:

```powershell
$remote = gh repo view imp-lesion-evidence-demo --json sshUrl -q .sshUrl
git remote add origin $remote
```

- [ ] **Step 5: Verify laptop clean clone**

Run on laptop:

```powershell
$remote = gh repo view imp-lesion-evidence-demo --json sshUrl -q .sshUrl
git clone $remote E:\imp-lesion-evidence-demo
Set-Location E:\imp-lesion-evidence-demo
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
git status --short
```

Expected: tests pass, paper builds, `git status --short` emits nothing.

- [ ] **Step 6: Commit**

```powershell
git add scripts/bootstrap_windows.ps1 docs/runbooks/two-machine-delivery.md .github/workflows/ci.yml README.md
git commit -m "docs: add two-machine delivery workflow"
git push origin main
```

### Task 13: Deploy, Verify, and Package Delivery

**Files:**
- Create: `scripts/demo/run_demo.ps1`
- Create: `scripts/demo/run_tunnel.ps1`
- Create: `demo/README.md`
- Create locally only: `demo_runtime/final_verification.json`

**Interfaces:**
- Produces: local URL, temporary public URL, final PDF, verification receipt, clean Git state.

- [ ] **Step 1: Write launch scripts**

`run_demo.ps1` checks all model/prior/evidence hashes before launching `127.0.0.1:7860`. `run_tunnel.ps1` checks local health then invokes:

```powershell
cloudflared tunnel --url http://127.0.0.1:7860
```

No tunnel token or generated URL is written into tracked files.

- [ ] **Step 2: Run the full verification suite**

Run:

```powershell
uv run --extra dev --extra analysis --extra demo pytest tests/demo -v
uv run --extra train --extra analysis --extra demo python -m lesion_robustness.demo.model_service --smoke-image demo_runtime/smoke.jpg
uv run --extra analysis python scripts/paper/build_clean_v3_tables.py --registry demo/data/evidence_registry.json --paper-dir paper/clean_v3_loop206
uv run --extra analysis python scripts/paper/audit_clean_v3_paper.py --paper paper/clean_v3_loop206 --registry demo/data/evidence_registry.json --receipt demo_runtime/paper_audit.json
Push-Location paper/clean_v3_loop206
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

Expected: all tests pass; GPU smoke passes; audit passes; PDF compiles.

- [ ] **Step 3: Rehearse browser states**

Verify desktop/mobile, no-GT, valid-GT, invalid mask, oversized image, checkpoint failure, evidence tab, legacy tab, and downloadable receipt. Invoke `$design-review` after the final screenshot set.

- [ ] **Step 4: Rehearse public tunnel**

Start demo and tunnel. From a separate browser session, complete one real comparison. Stop the tunnel after rehearsal. Record URL availability, timestamps, model hashes, prior hash, latency, and cleanup result in untracked `demo_runtime/final_verification.json`.

- [ ] **Step 5: Run completion gates**

Invoke `$ship-it-verified`, `$verification-before-completion`, `$requesting-code-review`, and `$constitutional-review`. Fix every correctness, evidence, privacy, or unsupported-claim finding.

- [ ] **Step 6: Final commit and push**

```powershell
git add scripts/demo/run_demo.ps1 scripts/demo/run_tunnel.ps1 demo/README.md paper/clean_v3_loop206/main.pdf
git commit -m "release: package evidence-first paper demo"
git push origin main
git status --short --branch
```

Expected: branch synchronized with `origin/main`; no tracked changes.

---

## 35-Hour Work Allocation

| Window | Main workstation | RTX 4060 laptop |
|---|---|---|
| 0-3 h | Tasks 1-3, GitHub private remote | Clone, bootstrap, CPU test proof |
| 3-10 h | Tasks 4-6, exact prior and GPU inference | Evidence registry review, table review |
| 10-18 h | Task 7, live app and local visual review | Tasks 8-9, paper prose and LaTeX |
| 18-24 h | Tasks 10-11, figures and integration | Citation, language, caption, claim audit |
| 24-30 h | Task 12, clean-clone and CI fixes | Independent paper build and UI smoke |
| 30-35 h | Task 13, tunnel rehearsal and release | Final PDF proofread and remote verification |

## Stop Conditions

- Dataset identity cannot be restored for all 384 Loop206 rows.
- Deployment prior does not reproduce all 76 frozen holdout contours byte-for-byte.
- Either selected checkpoint fails strict load or hash verification.
- Demo emits metrics without ground truth.
- Paper audit finds protected-test, SOTA, legacy-ranking, or unsupported statistical language.
- Public deployment exposes local paths, uploads, secrets, weights, or dataset files.

If either data or parity stop condition occurs, arbitrary-upload candidate inference stays disabled. Delivery continues with the complete paper, benchmark evidence website, and an explicit blocker report; no approximate candidate channel is substituted.
