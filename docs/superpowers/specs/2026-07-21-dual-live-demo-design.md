# Dual-Live Research Demo Design

**Status:** Approved direction; implementation requires written-spec review.

## Goal

Demonstrate that the trained IMP segmentation model and the recorded nnU-Net v2
baseline both perform real inference on the same dermoscopy image during a live
research defense. Expose one guarded Gradio URL through Cloudflare Quick Tunnel
without weakening the evidence contract, leaking local paths, or presenting the
output as clinical or protected-test evidence.

## Audience Outcome

By the end of the demo, an examiner should understand that two hash-bound model
artifacts executed on the submitted image, see how their masks differ, and know
which comparisons are illustrative versus scientifically scored.

## Evidence Boundaries

- Arbitrary uploads support live IMP and live nnU-Net inference.
- Arbitrary uploads do not display Dice, BF1, IoU, superiority, or significance
  because no provider-authorized ground truth exists.
- Fixed allowlisted train-screen samples may display provider-bound ground truth
  and metrics under their existing evidence class.
- Protected test-v3 remains sealed.
- Output is non-clinical research evidence and is not a diagnosis.
- The UI must never substitute a cached nnU-Net mask when it labels an output as
  live. A sidecar failure clears the nnU-Net panel and fails closed.

## Visual Language

Preserve the existing forensic research-instrument direction while fixing the
current contrast defect.

- Palette: ivory background, graphite evidence surfaces, teal for authorized
  actions, rust for constraints and warnings.
- Typography: expressive slab/serif display title with compact technical body
  copy; maintain a clear type scale and WCAG AA contrast.
- Layout: one primary workflow, not a card dashboard. The live comparison is the
  first screen; evidence and legacy audit remain secondary tabs.
- Motion: short result reveal only; honor `prefers-reduced-motion`.
- States: loading, model-ready, sidecar-unavailable, invalid upload, OOM, timeout,
  and cleared-output states receive explicit treatments.

## Primary Screen

The `Live Dual-Model Compare` screen contains:

1. A public sample selector plus an arbitrary dermoscopy upload control.
2. One `Run both models` action.
3. A single evidence strip showing input SHA-256, image dimensions, and execution
   mode.
4. Three aligned visual columns: original image, IMP mask/overlay, and nnU-Net
   mask/overlay.
5. A result ledger showing per-model latency, model ID, checkpoint SHA-256,
   preprocessing contract, output-mask SHA-256, and execution status.
6. A downloadable path-free JSON receipt.
7. A persistent non-clinical notice and evidence-class label.

The fixed-cache Loop206 comparison remains available as an audited secondary
workflow. Its candidate cache is not presented as live arbitrary-upload output.

## Runtime Architecture

### Windows Gradio Host

- Binds to `127.0.0.1:7860`.
- Owns uploads, image validation, the IMP model, presentation, receipt creation,
  and the public queue.
- Runs one inference job at a time.
- Keeps only the live IMP model resident. Loop206 candidate output remains a
  fixed-cache workflow unless a separate receipt explicitly authorizes it.

### WSL nnU-Net Sidecar

- Binds to `127.0.0.1:7862`; never binds externally.
- Loads the exact Loop192 checkpoint, plans, fingerprint, trainer, and dataset
  metadata whose hashes match the evidence report.
- Accepts one validated RGB image and a request ID over localhost.
- Applies the recorded raw-RGB 256x256 nnU-Net pipeline.
- Returns a binary mask, model ID, checkpoint SHA-256, output SHA-256, latency,
  and protocol identifier.
- Rejects requests when artifact hashes, environment identity, input limits, or
  output schema do not match.

### GPU Scheduling

- The Gradio queue admits one request.
- IMP inference runs first; nnU-Net runs second.
- A CUDA preflight measures resident memory and one full dual inference before
  public launch.
- The launch aborts on OOM, non-finite output, timeout, or unexpected device.
- No automatic CPU fallback is allowed for the public defense URL.

## WSL Recovery

The current `Ubuntu-E` distribution cannot resolve its recorded users and emits
filesystem/user database errors. Recovery is staged and non-destructive:

1. Record distro registration, VHD location, size, free disk space, and current
   WSL status.
2. Stop the distribution before any VHD operation.
3. Create a verified backup using `wsl --export --vhd` when possible. If export
   cannot start, copy the stopped VHD byte-for-byte to a separate backup path and
   verify size plus SHA-256.
4. Never unregister, reset, compact, overwrite, or delete the source distro.
5. Inspect the backup or a read-only attachment first. Recover the Loop192
   checkpoint, plans, fingerprint, trainer metadata, and environment lock data.
6. Compare every recovered artifact against the hashes recorded in the Loop192
   report. Stop on any mismatch.
7. Prefer restoring the existing environment. If it is unrecoverable, build a
   new isolated WSL sidecar environment while preserving the recovered artifacts
   and recorded inference contract.

## Request Flow

1. Validate upload type, decoded size, pixel count, and RGB conversion.
2. Snapshot immutable input bytes and compute the public input digest.
3. Run IMP inference and validate its binary output.
4. Send the same decoded RGB image to the localhost nnU-Net sidecar.
5. Validate the sidecar response, request ID, model ID, checkpoint digest, mask
   dimensions, and mask digest.
6. Resample both masks only for presentation on the original input canvas.
7. Build overlays and a path-free receipt.
8. Render both outputs together. If either live arm fails, do not show stale
   output from that arm.

## Error Handling

- Sidecar unavailable: disable dual-run action; explain that nnU-Net is offline.
- IMP failure: clear both result panels and return one sanitized error.
- nnU-Net failure after IMP success: show IMP as completed, clear nnU-Net, label
  the run incomplete, and disable receipt download.
- OOM or timeout: abort the public run, clear affected outputs, and preserve no
  upload outside the launcher-owned session.
- Never expose stack traces, local paths, WSL usernames, environment variables,
  cache locations, or checkpoint locations in the public UI or receipt.

## Cloudflare Deployment

- Start and verify the sidecar first.
- Start the guarded Gradio launcher and require local HTTP health plus one full
  dual-model smoke inference.
- Tunnel only `http://127.0.0.1:7860` using the existing Cloudflare wrapper.
- Treat the Quick Tunnel URL as temporary and unauthenticated.
- Use only synthetic or already-public rehearsal images.
- Stop the tunnel first, then the Gradio host, then the sidecar. Verify ports
  `7860` and `7862` are closed and launcher-owned temporary uploads are removed.

## Verification

- Unit tests for sidecar request/response schemas, hash pins, image limits,
  timeout, malformed output, and path sanitization.
- Service tests proving both live models execute and cached output cannot satisfy
  a live request.
- Receipt tests binding input, outputs, model IDs, checkpoint hashes, protocol
  identifiers, and latencies without local paths.
- CUDA preflight on the RTX 5060 Ti and later the RTX 4060 laptop.
- Browser checks at 1440x900 and 390x844 for contrast, layout, loading, success,
  sidecar-error, and upload-error states.
- Public URL smoke test from outside localhost before handoff.

## Non-Goals

- No clinical diagnosis, calibration claim, protected-test claim, or SOTA claim.
- No concurrent multi-user GPU serving.
- No permanent Cloudflare DNS or authenticated production deployment in this
  iteration.
- No repair action that risks deleting or overwriting the existing WSL distro.
