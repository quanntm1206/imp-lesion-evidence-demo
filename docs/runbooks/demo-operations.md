# Dual-Model Demo Operations

## Scope And Evidence Limits

NON-CLINICAL RESEARCH DEMO. Outputs are segmentation visualizations, not a
diagnosis, clinical decision support, medical-device output, or proof that one
model is superior.

The primary live path decodes one RGB image, runs IMP first, then the
reconstructed nnU-Net runtime on the same contiguous RGB array. The runtime is
reconstructed from pinned artifacts and dependencies. It is not claimed to be
equivalent to the unavailable original nnU-Net environment.

The current browser E2E receipt is absent and unverified in this release;
browser rendering and desktop/mobile screenshots remain unverified.

- Public fixed samples are the preferred defense path.
- Arbitrary uploads are `Exploratory - no ground truth`; show no accuracy,
  Dice, IoU, HD95, ASSD, or comparative-performance claim.
- Audited metrics belong only in the separate fixed-sample evidence tab.
- `VAL_GATE_FAILED_NO_TEST` means protected test-v3 remains sealed. Never
  relabel train-screen or fixed-cache evidence as protected-test evidence.
- Receipts contain current live hashes, model/checkpoint identity, geometry,
  latency, and device only. They contain no local path, username, diagnosis,
  or accuracy for an arbitrary upload.

## Required Local Inputs

Keep all values outside source control, screenshots, receipts, and tunnel
commands:

- `IMP_LOOP206_CONTROL_CHECKPOINT`
- `IMP_LOOP206_CANDIDATE_CHECKPOINT`
- `IMP_LOOP206_DATA_ROOT`
- `$PythonExe`, set to the local Python executable containing the pinned demo
  dependencies; the guarded launcher accepts it with `-PythonExe` and records
  the resolved executable identity without storing this private path in Git
- verified Loop192 bundle under
  `demo_runtime/nnunet/recovered-container-final2`, or
  pass its private path with `run_sidecar.ps1 -BundlePath`

The release registry intentionally has no approved `prior_receipt_sha256`.
Do not set `IMP_LOOP206_PRIOR` or `IMP_LOOP206_PRIOR_RECEIPT`; the guarded
launcher clears them.

## One-Time Recovery And Image Build

Recovery is a private, local operation. Open an elevated PowerShell window.
Set `IMP_NNUNET_VHD` to the private source VHD and `IMP_NNUNET_REPORT` to the
pinned Loop192 report. The output directory must not already exist or must be
empty:

```powershell
if ([string]::IsNullOrWhiteSpace($env:IMP_LEGACY_LINUX_USER)) {
  throw 'Set IMP_LEGACY_LINUX_USER to the source VHD account name.'
}
$outputRoot = Join-Path (Resolve-Path '.').Path 'demo_runtime/nnunet/recovered-container-final2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/recover_nnunet_artifacts.ps1 `
  -VhdPath $env:IMP_NNUNET_VHD `
  -ReportPath $env:IMP_NNUNET_REPORT `
  -OutputRoot $outputRoot `
  -PythonExe '.venv-win\Scripts\python.exe' `
  -LegacyLinuxUser $env:IMP_LEGACY_LINUX_USER
```

Require `recovery=passed`, exact artifact hashes, `source_vhd_unchanged=true`,
and cleanup proof that the filesystem is unmounted and the VHD detached. Stop
on any cleanup warning; do not reuse a partial output directory. Then require
`recovery_receipt.json` SHA-256
`0470993eeea5fd39a970400af2465a3f43cfbd1c1bb75ccda2202ef5de362a77`.
Do not push the recovered bundle, source VHD, checkpoint, dataset, or receipt.

Build the pinned reconstructed runtime from the repository root:

```powershell
docker build -t imp-nnunet-sidecar:loop192 -f sidecar/nnunet/Dockerfile .
```

The launcher requires image ID
`sha256:86bd77c03c3918e3638565e29417cdf4360b499a0813fbc425dc36645f026f2d`.
Stop if the build does not match. Do not weaken the pin.

## Guarded Start Order

Use three PowerShell terminals. Required order: sidecar, Gradio, Cloudflare.
Choose one safe, previously unused public run ID before starting the sidecar:

```powershell
$RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffffffZ').ToLowerInvariant()
```

1. Validate hashes, Docker GPU access, checkpoint load, CUDA/API identity, and
   sidecar health without leaving a container running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -CheckOnly -PreserveMode -RunId $RunId
```

This creates a unique launcher-generated, owner-bound container name, then
stops that exact container after pinned health succeeds. Preserve mode must
retain and re-inspect the exact container ID, name, and owner with
`State.Running=false`; absence is a failure. Require port `7862` to be closed
and immutable owner/start/stop journals to remain. Absence is accepted only for
non-preserve auto-remove. The stopped preserved container name is intentionally
not reused or removed.

2. Start the persistent sidecar. It publishes only
   `127.0.0.1:7862:7862`, mounts the model read-only, uses one GPU, and writes
   a launcher-owned record:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -PreserveMode -RunId $RunId
```

3. In terminal 2, run the guarded Gradio preflight. It requires exact sidecar
   identity and a complete CUDA dual-model
   smoke inference before binding:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -CheckOnly -PublicTunnelMode -PreserveMode -RunId $RunId -PythonExe $PythonExe
```

Require `preflight=passed` and `dual_smoke=passed`. Then start Gradio:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -PublicTunnelMode -PreserveMode -RunId $RunId -PythonExe $PythonExe
```

Confirm the app is available only at `http://127.0.0.1:7860`. Run two
materially different public samples. Both live arms must complete; mask hashes
must change across inputs. Repeat one sample to record same-current-runtime
determinism only, not original-runtime equivalence.

4. In terminal 3, start a temporary Cloudflare Quick Tunnel only after local
   inference passes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_tunnel.ps1 -PreserveMode -RunId $RunId
```

The wrapper accepts only the preserved public-mode Gradio journal with the
current release-manifest digest, then verifies launcher/process/session/config
identity. Cloudflare exposes only `http://127.0.0.1:7860`. Port 7862 remains
local.

## Public Rehearsal

The Quick Tunnel is temporary and unauthenticated. Its public mode renders and
accepts only bundled public or synthetic inputs; no upload control or upload API
is exposed, and forged upload callbacks are rejected server-side. Never use
patient, identifying, confidential, or unpublished images. The tunnel itself is
not an access-control boundary.

From a separate network, load the temporary URL and run one approved public
sample. Verify Original, IMP, and nnU-Net panels all correspond to the current
request. Do not publish the URL, terminal output, upload, receipt, or screenshot
containing the URL.

## Failure Behavior

- If IMP fails, show neither model result and create no receipt.
- If nnU-Net fails after IMP, the current IMP result may remain visible;
  clear nnU-Net output and receipt. Never reuse a stale nnU-Net mask.
- Reject malformed JSON, request/hash/model/checkpoint/geometry drift, CPU
  fallback, oversized input, and non-binary/non-finite masks.
- Show a sanitized user-facing error. Never show traceback, Docker log, local
  path, cache, checkpoint location, or prior request output.
- Keep the app loopback-only when any preflight, ownership, evidence, or
  outside-network smoke check is unavailable.

## Ordered Shutdown And Cleanup Proof

Use the launcher-owned stop script. It attempts every step, aggregates errors,
and enforces this order: Cloudflare, Gradio, sidecar.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/stop_demo.ps1 -PreserveMode -RunId $RunId
```

Require successful cleanup output and confirm ports 7860 and 7862 are closed.
The named container must be absent. Preserve-mode journals, receipts, caches,
sessions, and containers are retained as append-only evidence; unrelated runtime
files must remain. Treat any nonzero exit as an incomplete shutdown and do not
reopen the tunnel until resolved.
