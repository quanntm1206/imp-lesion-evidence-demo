# Dual-Model Evidence Demo

This non-clinical research demo has two separated surfaces:

- `Live Dual-Model Compare`: one RGB input runs sequentially through IMP, then
  reconstructed-runtime nnU-Net. It shows Original, IMP, and nnU-Net masks.
- `Audited Fixed Samples`: provider-authorized fixed evidence. The historical
  comparison class remains
  `train_screen / exact_fixed_cache / historical_cache_provenance_drift`.

Arbitrary upload is exploratory: no ground truth, no accuracy, and no clinical
or state-of-the-art claim. Earlier arbitrary-upload candidate authorization was
control-only because exact prior parity was `0/76`; the live nnU-Net arm is a
separate reconstructed runtime with pinned artifacts, not a substitute prior.

## Local Launch

Run from the repository root with CUDA. Start the sidecar first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1
```

Then validate the complete sequential smoke and start Gradio on
`127.0.0.1:7860`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1
```

The guarded launcher validates evidence/model/checkpoint hashes, exact sidecar
identity, CUDA device, live preprocessing, and one complete dual inference
before public startup. Queue concurrency is one. CPU is for explicit local
rehearsal only; public dual-live launch requires CUDA.

The current browser E2E receipt is absent and unverified in this release;
browser rendering and desktop/mobile screenshots remain unverified.

## Temporary Public Rehearsal

Start Cloudflare only after local dual inference passes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_tunnel.ps1
```

The wrapper exposes only Gradio and runs the equivalent target:

```text
cloudflared tunnel --url http://127.0.0.1:7860
```

The generated URL is temporary and unauthenticated. Use only non-sensitive
synthetic or already-public rehearsal images. Assume submitted bytes may reach
the operator host. Never use clinical, identifying, confidential, or
unpublished images.

Each app launch scopes temporary upload storage to a unique launcher-owned
session under `demo_runtime/sessions`. It is removed automatically when the app
stops. A containment check protects every sibling runtime file and global temp
directory. Cleanup failure exits nonzero.

Stop all resources in Cloudflare, Gradio, sidecar order:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/stop_demo.ps1
```

Do not rely only on `Ctrl+C`; require the stop script to prove ports 7860 and
7862 are closed.

## Evidence Limits

- Uploaded-image output is illustrative and unscored.
- The provider-bound train-screen GT checkbox enables metrics only for the
  current audited fixed result. User-supplied ground truth never enables them.
- Protected test-v3 remains sealed; `VAL_GATE_FAILED_NO_TEST` is preserved.
- Current-runtime determinism does not prove original-runtime equivalence.
- Invalid masks, stale bindings, oversized images, checkpoint failures, or an
  unavailable arm fail closed without local paths, environment names, upload
  names, stack traces, cache locations, or weight locations.
- If nnU-Net fails after IMP, current IMP output may remain; nnU-Net and receipt
  clear. No incomplete live run produces a receipt.

Full startup, public-smoke, failure, and two-machine procedures live in
`docs/runbooks/demo-operations.md` and
`docs/runbooks/two-machine-delivery.md`.

For a teacher/operator starting from a clean machine, use
[`DEMO_DEPLOYMENT_GUIDE.md`](../DEMO_DEPLOYMENT_GUIDE.md). It lists the
out-of-band artifact package, SHA-256 verification, exact commands,
acceptance checks, troubleshooting, and ordered shutdown.
