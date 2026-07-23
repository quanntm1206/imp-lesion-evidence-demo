# Continue Later

This branch is the resumable, public-safe development checkpoint for the IMP
paper, presentation, demo, and prospective RQ1-v2 work.

## Restore the repository

New machine:

```powershell
git clone https://github.com/quanntm1206/imp-lesion-evidence-demo.git
Set-Location imp-lesion-evidence-demo
git switch --track -c continue origin/continue
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
.venv-win\Scripts\python.exe -m pytest -q
```

Existing clone:

```powershell
git fetch origin
git switch continue
git pull --ff-only origin continue
```

The clean submission base is tag `submission-2026-07-23-v3`. The `continue`
branch adds continuation notes and retained visual-QA evidence without changing
the bounded scientific claims in that release.

## What GitHub restores

- Application, sidecar, launcher, research-contract, and verification source.
- Paper source and PDF; interactive HTML, PPTX, PDF, and presentation source.
- Vietnamese presentation script and lecturer Q&A.
- Demo deployment guide, model card, data card, reproducibility notes, and
  two-machine runbook.
- Six prospective RQ1-v2 configs and the honest `pending/unverified` result
  manifest.
- Slide render evidence under `outputs/visual-evidence/`.

## What GitHub does not restore

The public branch intentionally excludes all model/data/runtime payloads. A
pull alone cannot run real inference or resume training. Before deleting the
current machine, make a separate hash-verified backup containing:

- IMP Loop206 control and candidate checkpoints.
- Loop206 prior, prior receipt, fixed caches, source reports, and dataset index.
- Public sample image bytes referenced by the dataset index.
- Reconstructed nnU-Net bundle, `checkpoint_final.pth`, dataset metadata,
  recovery receipt, and the pinned Docker image or its verified archive.
- Authorized Clean-v3 index/data and model initialization artifacts if the
  prospective RQ1-v2 study will be run later.

Use `docs/runbooks/two-machine-delivery.md` to create and verify the path-safe
`sha256-manifest.json`. Restore the private roots exactly as described in
`DEMO_DEPLOYMENT_GUIDE.md`; never commit them to GitHub.

## Resume order

1. Restore this branch and run the CPU test suite.
2. Restore the private artifact bundle; verify its recursive SHA-256 manifest.
3. Restore/build the pinned nnU-Net Docker image; run CUDA and launcher
   preflights before starting the demo.
4. Re-run local dual-model, browser, and Cloudflare smoke checks. Treat every
   Quick Tunnel URL as temporary.
5. Resolve the authorized Clean-v3 index gate: exactly 2,008 train and 431
   validation records; keep test-v3 and PH2 sealed.
6. Implement the real tracked training/evaluation engines. Current RQ1-v2
   entrypoints are contract-only and fail closed on missing private inputs.
7. Run exactly six jobs: IMP and nnU-Net at seeds 206, 1206, and 2206.
8. Promote results only after all six receipts and per-case artifacts validate;
   then compute the nine contrasts, uncertainty intervals, and registered
   slices before updating the paper.

Current prospective status: `pending/unverified`, `0/6` completed jobs,
`p1_status=not_promoted`. Do not create metrics or learning curves until real
jobs validate.

## Deliberately omitted scratch

Agent plans, `.superpowers`, temporary test fixtures, virtual environments,
LaTeX intermediates, caches, uploads, raw receipts, tunnel logs, and generated
base/inspection files are not continuation inputs. Regenerate them when needed.

## Repository follow-ups

- Choose and add a license before inviting external reuse.
- Decide separately whether to rewrite public Git history and remove stale
  branches that retain older internal artifacts. Do not force-push casually.
- Keep `main` as the clean submission surface; do future work on `continue` or
  a branch created from it.
