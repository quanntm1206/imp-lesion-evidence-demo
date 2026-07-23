# Reproducibility Status

## What this repository supports

This release is a reproducibility scaffold. A clean clone can inspect the
research protocol, run portable tests, validate the evidence registry, rebuild
manifest-bound tables, and compile the manuscript. It cannot reproduce the
historical training runs from tracked files alone.

The repository keeps four evidence lanes separate: Historical Paper RQ1,
Loop206 train-screen ablation, Live reconstructed runtime, and Prospective
RQ1-v2. Validation within one lane does not promote evidence into another.

## Verification levels

**Registry-only verification** checks registry semantics, expected hashes,
citations, claims, and every source byte that is present. Missing private source
reports remain explicit warnings. Registry-only success is not full experiment
reproduction and must not be relabeled as strict.

**Strict local audit** runs only where all registry source reports are available
and fails on a missing or mismatched source byte:

```powershell
.venv-win\Scripts\python.exe scripts/paper/audit_clean_v3_paper.py `
  --paper paper/clean_v3_loop206 `
  --registry demo/data/evidence_registry.json `
  --receipt .artifacts/paper-audit-strict.json
```

The receipt is local evidence. Do not commit it when it contains private
runtime context.

## Historical limitations

The tracked release does not contain the complete historical Loop191/192
training implementation/config closure, all six Loop206 configurations,
checkpoints, paired predictions, probability caches, or compact source reports.
The Clean-v3 split is therefore **not independently reconstructable** from this
commit even though the historical recorded audit and aggregate metadata remain
available for review.

The reconstructed nnU-Net service binds current dependencies, model identity,
checkpoint identity, geometry, and request/output hashes. Those bindings do not
prove original-runtime equivalence. Live masks have no ground truth and support
no accuracy or clinical claim.

## Prospective RQ1-v2 gate

RQ1-v2 remains **pending/unverified**. Promotion requires an authorized,
hash-pinned Clean-v3 index; a passing integrity report; six locked job configs;
all **six jobs** (two systems times three seeds); job receipts; and the
prespecified aggregation contract. The current protocol deliberately blocks
before data access because the authorized prospective index is unresolved.

The contract-only train/evaluate/summarize entrypoints provide `-PreflightOnly`
and `-DryRun` modes. Missing or mismatched private prerequisites fail closed
with **Exit code 2** before training, evaluation, dataset traversal, or metric
output. No metric table or learning curve may be created until all six jobs
validate and the result manifest is promoted from `pending/unverified`.

## Private artifact transfer

**Private artifact transfer** uses LAN or removable media, never GitHub. Keep
datasets, masks, checkpoints, caches, raw receipts, Docker/VHD archives, and
credentials outside the repository. Generate a relative-path SHA-256 manifest
at the source, recompute it at the destination, and reject missing, additional,
renamed, or changed files. Follow
[`runbooks/two-machine-delivery.md`](runbooks/two-machine-delivery.md) for the
path-safe transfer procedure.

## Portable checks

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
.venv-win\Scripts\python.exe -m pytest -q
.venv-win\Scripts\python.exe scripts/paper/audit_clean_v3_paper.py `
  --paper paper/clean_v3_loop206 `
  --registry demo/data/evidence_registry.json `
  --receipt .artifacts/paper-audit-registry-only.json `
  --source-verification registry-only
```

A zero exit code establishes only the checks named by that command. Hardware,
browser, Cloudflare, CUDA, strict-source, and six-job evidence require their own
receipts.
