# Loop206 Demo Operations

## Scope Warning

NON-CLINICAL RESEARCH DEMO. Segmentation output is not a diagnosis, clinical
decision support, or medical-device output. Do not use it for patient care.

## Required Environment Variables

Set these names through the approved local secret manager or service wrapper.
Do not put values in this runbook, source control, screenshots, receipts, or
tunnel commands.

- `IMP_LOOP206_CONTROL_CHECKPOINT`
- `IMP_LOOP206_CANDIDATE_CHECKPOINT`
- `IMP_LOOP206_DATA_ROOT`

The current release registry has no approved `prior_receipt_sha256`. Do not set `IMP_LOOP206_PRIOR` or `IMP_LOOP206_PRIOR_RECEIPT`; configured values are rejected before receipt parsing or deserialization. The guarded public launcher also clears both names.

## Preflight

1. Verify the evidence registry semantic hash. Require it to match
   `artifact_manifest.json`, fixed-cache receipts, and the paper-audit receipt.
2. Verify control checkpoint, candidate checkpoint, candidate-cache manifest,
   and zero-cache manifest SHA-256 values against the approved model registry
   and fixed-cache receipt.
3. Confirm the release registry leaves `prior_receipt_sha256` unset and the
   arbitrary-upload candidate remains disabled. A future release may enable it
   only by pinning an approved passed-receipt digest after exact 76/76 parity;
   do not substitute an adjacent self-hash or approximate prior.
4. Run the paper audit. It must report `passed=true errors=0` before release.

## Local Start And Health

Browser rendering and desktop/mobile screenshots remain unverified. This section defines the required operator verification; it does not record that verification as complete.

1. Start `scripts/demo/run_demo.ps1` on the loopback interface only. Direct
   `lesion-demo` launch is rejected because it lacks the launcher-owned upload
   session guard; direct `lesion-demo --share` is always rejected.
2. Open the local workbench. Confirm the degraded-runtime banner, evidence
   registry hash, pinned model hashes, exact fixed-cache selector, and
   non-clinical warning render.
3. Run one allowlisted fixed-cache comparison without ground truth. Confirm no
   Dice, IoU, boundary, HD95, or ASSD metrics are shown.
4. Run the same approved sample with provider-bound ground truth. Confirm both
   model hashes, the evidence badge, and a path-free JSON receipt.
5. Verify queue concurrency remains `1`; do not add workers or a parallel
   inference queue.

## Upload Handling

- Treat uploads as ephemeral control-only preview inputs.
- The server rejects uploads above 16 MiB, and Pillow rejects decoded images
  above 16 megapixels before Gradio image conversion.
- Clear temporary uploads and generated receipt files after each session and
  after every failed request.
- Never retain raw uploads, masks, output arrays, or absolute filesystem paths
  in logs, receipts, browser downloads, or tunnel diagnostics.
- Candidate output for arbitrary uploads remains disabled unless exact prior
  parity is re-established through the approved evidence workflow.

## Tunnel Procedure

1. Complete local health checks first.
2. Start the approved tunnel service through its managed configuration; do not
   place credentials, public addresses, or tokens in shell history or logs.
3. Recheck the external surface: non-clinical warning, candidate lockout,
   fixed-cache-only comparison, one-worker queue, and path-free receipts.
4. Stop the tunnel immediately after review. Confirm no public listener remains
   and delete transient tunnel logs containing request metadata.

## Failure Recovery

1. Stop the service and tunnel when a hash, prior parity, cache, registry, or
   audit check fails.
2. Preserve only the path-free failure receipt and command exit status.
3. Restore the last approved immutable model, prior, cache, registry, and
   manifest set. Re-run preflight and local health checks before restart.
4. If candidate authorization fails, restart only in control-preview plus exact
   fixed-cache mode. Do not bypass the lockout.

## Local-Only Fallback

Keep the service loopback-only when tunnel approval, health verification, or
any evidence check is unavailable. Operators may review fixed-cache evidence
locally; they must not expose an unverified service publicly.
