# Evidence-First Demo Operations

This non-clinical research demo exposes two deliberately different modes:

- Exact Fixed-Cache Compare: the fixed allowlist of 76 train-screen samples supports the exact dual comparison class `train_screen / exact_fixed_cache / historical_cache_provenance_drift`.
- Arbitrary upload: control-only. Candidate inference remains locked because exact prior parity is `0/76`; no serialized prior, zero contour, or approximate contour is substituted.

## Local launch

From the repository root, launch with the existing CUDA overlay. The script uses `.venv-win\Scripts\python.exe` directly, validates the evidence registry semantic hash, pinned model bindings, checkpoint hashes, fixed-cache manifest and mmap hashes, live preprocessing config, and dataset-index hash before Gradio binds to `127.0.0.1:7860`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1
```

Use `-Device cpu` only for a deliberate CPU rehearsal. Use `-CheckOnly` for preflight without opening the server. The queue has one worker.

## Temporary public rehearsal

Install `cloudflared` on the operator machine. Start the local demo first. In a second terminal run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo/run_tunnel.ps1
```

The wrapper checks `http://127.0.0.1:7860`, resolves the executable as an application, then runs:

```text
cloudflared tunnel --url http://127.0.0.1:7860
```

The generated URL is temporary. Never add the URL, tunnel output, tokens, uploads, caches, weights, or runtime receipts to Git. Complete the authorized rehearsal, then press `Ctrl+C` in the tunnel terminal. Press `Ctrl+C` in the demo terminal afterward. Confirm port 7860 and both processes are closed.

The quick tunnel is unauthenticated. Use only non-sensitive synthetic or already-public rehearsal images. Assume a submitted image may reach the operator host. For every app launch, the launcher creates a unique launcher-owned child under `demo_runtime/sessions` and scopes all Gradio temporary upload storage to that child. The prior environment is restored and the child is removed automatically when the app stops. A containment check runs again immediately before deletion; sibling runtime files and global temporary storage are never removed. Cleanup failure exits nonzero. Never reuse clinical, identifying, or confidential images.

## Evidence limits

Outputs are illustrative train-screen evidence. They are not validation, protected-test, clinical, or state-of-the-art claims. A ground-truth upload enables metrics only for the current fixed result. Invalid masks, oversized images, checkpoint failures, and unavailable candidate state fail closed without showing local paths, environment names, upload names, stack traces, cache locations, or weight locations in the public UI or downloadable receipt.
