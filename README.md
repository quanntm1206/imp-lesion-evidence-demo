# Evidence-Bounded Comparison of MiT-B3 U-Net and nnU-Net for Skin Lesion Segmentation

Research authors: Nguyễn Trần Minh Quân and Nguyễn Đức Lân.

This public repository is an evidence-bounded paper/demo package and a
**reproducibility scaffold**. It is not a clone-runnable training release. The
main research system is the **IMP MiT-B3 U-Net**: a MiT-B3 encoder coupled to a
U-Net decoder. The live comparison uses a **reconstructed nnU-Net** runtime.
That runtime is identity-pinned for the current demonstration but is not proven
equivalent to the original historical nnU-Net execution.

No repository artifact establishes clinical validity, state of the art, statistical superiority, or protected-test performance. Uploaded-image masks
are illustrative and unscored.

## Evidence lanes

Do not combine results across these four lanes:

1. **Historical Paper RQ1**: Loop191 IMP and Loop192 nnU-Net aggregate point
   estimates on adaptively used Clean-v3 development-validation under an older
   geometry contract; single run per architecture.
2. **Loop206 train-screen ablation**: matched zero-channel control versus a
   contour-channel candidate on 76 train-screen groups and three selected
   seeds; the negative interval is conditional on those seeds.
3. **Live reconstructed runtime**: L206 control followed by reconstructed L192
   nnU-Net on the same RGB input; no ground truth, live accuracy, or
   original-runtime equivalence claim.
4. **Prospective RQ1-v2**: a planned six-job comparison with original-image
   metric geometry. Its status is **pending/unverified** and no result is
   promoted.

The historical Paper RQ1, Loop206 evidence, fixed-sample cache, and live demo
remain audit records, not substitutes for prospective RQ1-v2 admission.

## Five-minute setup

Prerequisites: Git, `uv`, Python 3.12, and a TeX toolchain for the paper. From a
clean Windows clone:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
```

Run the complete tracked test suite:

```powershell
.venv-win\Scripts\python.exe -m pytest -q
```

Tests that need separately transferred runtime artifacts skip with
`external runtime assets; local release gate required`. PowerShell launcher
execution tests run on Windows; static launcher checks remain portable.

Rebuild the paper after the evidence audit passes:

```powershell
Push-Location paper/clean_v3_loop206
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

With separately transferred private checkpoints and a CUDA-capable Docker
runtime, check and launch the local demo in order:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1
```

Teacher/operator deployment, including the required private artifact layout,
hash verification, CUDA environment, exact guarded launch order,
troubleshooting, and shutdown proof: [`DEMO_DEPLOYMENT_GUIDE.md`](DEMO_DEPLOYMENT_GUIDE.md).

The bootstrap validates tracked artifacts; it does not reproduce historical
training. See [`docs/reproducibility.md`](docs/reproducibility.md) for the
verification levels and blocked RQ1-v2 gate.

## RQ1-v2 contract-only commands

These commands exercise a **contract-only scaffold**. They validate public
protocol/config structure and, in preflight mode, private prerequisite
bindings. They do not contain a training or evaluation engine. RQ1-v2 remains
**pending/unverified**.

Public dry runs open no private data and write no checkpoint or metric:

```powershell
.venv-win\Scripts\python.exe scripts/research/train_rq1_v2.py --protocol experiments/rq1_v2/protocol.json --config experiments/rq1_v2/configs/imp_seed206.yaml --dry-run
.venv-win\Scripts\python.exe scripts/research/evaluate_rq1_v2.py --protocol experiments/rq1_v2/protocol.json --config experiments/rq1_v2/configs/imp_seed206.yaml --dry-run
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research/reproduce_paper_results.ps1 -DryRun
```

Private preflight checks require the authorized index, experiment manifest,
input artifact, expected SHA-256, and output root to be supplied out of band:
`IMP_RQ1_V2_EVAL_CHECKPOINT` and its SHA-256 must identify the trained output
being evaluated, not the initialization artifact.

```powershell
$ImpOutput = Join-Path $env:IMP_RQ1_V2_OUTPUT_ROOT 'imp_seed206/final.pt'
$EvalCheckpoint = $env:IMP_RQ1_V2_EVAL_CHECKPOINT
$EvalCheckpointSha256 = $env:IMP_RQ1_V2_EVAL_CHECKPOINT_SHA256
.venv-win\Scripts\python.exe scripts/research/train_rq1_v2.py `
  --protocol experiments/rq1_v2/protocol.json `
  --config experiments/rq1_v2/configs/imp_seed206.yaml `
  --data-manifest $env:IMP_CLEAN_V3_INDEX `
  --experiment-manifest $env:IMP_RQ1_V2_EXPERIMENT_INPUT `
  --input-artifact $env:IMP_RQ1_V2_IMP_INITIALIZATION `
  --input-artifact-sha256 $env:IMP_RQ1_V2_IMP_INPUT_SHA256 `
  --output-checkpoint $ImpOutput `
  --preflight-only
.venv-win\Scripts\python.exe scripts/research/evaluate_rq1_v2.py `
  --protocol experiments/rq1_v2/protocol.json `
  --config experiments/rq1_v2/configs/imp_seed206.yaml `
  --data-manifest $env:IMP_CLEAN_V3_INDEX `
  --experiment-manifest $env:IMP_RQ1_V2_EXPERIMENT_INPUT `
  --input-artifact $EvalCheckpoint `
  --input-artifact-sha256 $EvalCheckpointSha256 `
  --preflight-only
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research/reproduce_paper_results.ps1 -PreflightOnly
```

The current unresolved prospective index makes private preflight fail closed
with exit code `2`; this is the expected pending state, not a scientific result.

## Repository map

| Path | Purpose |
|---|---|
| `src/lesion_robustness/` | Demo, evidence, metric, and currently tracked research components |
| `demo/` | Evidence registry and demo-facing metadata |
| `scripts/demo/` | Guarded sidecar, Gradio, tunnel, and shutdown launchers |
| `scripts/paper/` | Evidence audit and deterministic table/figure builders |
| `experiments/rq1_v2/` | Prospective protocol and canonical contract material |
| `paper/clean_v3_loop206/` | LaTeX manuscript and manifest-bound artifacts |
| `presentation/interactive/` | Interactive HTML deck source |
| `tests/` | Portable contract and behavior checks |

## Defense package

- Interactive deck: [`outputs/imp-lesion-evidence-defense.html`](outputs/imp-lesion-evidence-defense.html)
- PowerPoint: [`outputs/imp-lesion-evidence-defense.pptx`](outputs/imp-lesion-evidence-defense.pptx)
- Slide PDF: [`outputs/imp-lesion-evidence-defense.pdf`](outputs/imp-lesion-evidence-defense.pdf)
- Paper PDF: [`paper/clean_v3_loop206/main.pdf`](paper/clean_v3_loop206/main.pdf)
- Vietnamese talk script: [`docs/presentation/defense-presentation-script-vi.md`](docs/presentation/defense-presentation-script-vi.md)
- Lecturer Q&A: [`docs/presentation/lecturer-questions-and-answers-vi.md`](docs/presentation/lecturer-questions-and-answers-vi.md)
- Evidence status: [`reports/paper_revision/manuscript_readiness_audit.md`](reports/paper_revision/manuscript_readiness_audit.md)

## Artifact policy

Dataset bytes, masks, checkpoints, probability caches, source reports, uploads,
runtime receipts, Docker/VHD archives, credentials, and ephemeral tunnel URLs
stay **outside GitHub**. Transfer authorized private artifacts out of band and
verify SHA-256 before use. Never substitute absent artifacts with generated
numbers or placeholder model weights.

No public tunnel URL is pinned in this repository. A Quick Tunnel is temporary
and unauthenticated; use only public or synthetic images during rehearsal.

This tracked release exposes `lesion-demo` and `lesion-build-evidence` as
packaged entry points. Historical training code/config closure is incomplete;
the split identities cannot yet be independently reconstructed from a clean
clone. Details: [`docs/model-card.md`](docs/model-card.md),
[`docs/data-card.md`](docs/data-card.md), and
[`docs/runbooks/two-machine-delivery.md`](docs/runbooks/two-machine-delivery.md).

Browser rendering and desktop/mobile screenshots remain unverified. Tunnel
observations belong to their recorded runtime receipt; runbook instructions
alone are not release evidence.

See [`docs/runbooks/two-machine-delivery.md`](docs/runbooks/two-machine-delivery.md) for machine roles, private branch policy, clean-clone handoff, CI receipts, and hash-verified artifact transfer.

## RQ1-v2 data-integrity gate

RQ1-v2 uses only the ISIC 2016, 2017, and 2018 Task 1 image/mask material represented by the Clean-v3 manifest. Acquire those public challenge files from `https://challenge.isic-archive.com/data/` under their recorded licenses. Preserve the downloaded bytes. Verify every downloaded archive/file against the acquisition checksums before extraction, then verify the complete derived Clean-v3 manifest with:

```powershell
Get-FileHash -Algorithm SHA256 data/splits/clean_v3_manifest.csv
```

The required manifest digest is `4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102`. The authorized index schema is `imp.rq1_v2.dataset_index.v1`: top-level `clean_v3_manifest_sha256`, explicit `roots`, and `rows`; each row binds `sample_id`, `split`, `group_key`, `source_dataset`, root-relative image/mask references, `sha256_raw`, `sha256_rgb`, and `mask_sha256`. Only exact `train` and `validation` capabilities are accepted. Test-v3 denial occurs before index resolution. PH2 denial occurs before referenced file resolution.

Run the audit only after an authorized index exposes exactly 2,008 train rows and 431 validation rows. Change the protocol index status from `unresolved_blocked` to `verified` and record that index's real whole-file SHA-256; the reader verifies both before resolving any referenced file:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.venv-win\Scripts\python.exe scripts/research/audit_rq1_v2_data.py --protocol experiments/rq1_v2/protocol.json --index $env:IMP_CLEAN_V3_INDEX --output experiments/rq1_v2/data_integrity_report.json
```

The audit hashes canonical ASCII records `sample_id|group_key|sha256_raw|sha256_rgb\n`, sorted by sample ID and group key. It rejects cross-split group or decoded-RGB hash overlap. It also rejects a 63-bit luminance pHash candidate at Hamming distance at most 4 when luminance SSIM at 256x256 is at least 0.98. The immutable `imp.rq1_v2.data_integrity_report.v1` output contains no filesystem location and records `test_v3_open_count=0`.

Current readiness blocker: the locally discovered file with digest `e88a3cc144b799d214f40b85064665d3348bc8bac3ead549f80b96d436f69fc3` has schema `loop206.demo.dataset_index.v1` and only 308 train plus 76 train-screen-holdout rows. That digest remains valid only for its existing demo/public release role; it is not an RQ1-v2 index pin. The RQ1-v2 protocol records `dataset_index_status="unresolved_blocked"` and `dataset_index_sha256=null`, so the audit entry point rejects before opening an index or referenced data. Therefore `experiments/rq1_v2/data_integrity_report.json` remains intentionally absent; no zero-leakage result is claimed.
