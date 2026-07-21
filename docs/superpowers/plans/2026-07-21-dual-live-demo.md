# Dual-Live IMP + nnU-Net Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve one guarded Gradio URL that performs sequential live IMP and hash-pinned nnU-Net v2 inference on the same image, renders both outputs, and emits a path-free evidence receipt.

**Architecture:** The Windows Gradio host owns validation, IMP inference, presentation, receipts, and a single-worker queue on `127.0.0.1:7860`. A fresh Docker/WSL2 nnU-Net sidecar owns the recovered Loop192 model; its container listener uses `0.0.0.0:7862` only inside the isolated container namespace and Docker publishes it exactly as host `127.0.0.1:7862`. The original damaged `Ubuntu-E` VHD is attached only read-only for artifact extraction, never used at runtime. A versioned localhost JSON protocol carries a lossless RGB PNG plus hash bindings; all failures clear stale outputs and fail closed.

**Tech Stack:** Python 3.12, NumPy, Pillow, Gradio 5/6, stdlib HTTP/JSON, PyTorch CUDA, nnU-Net v2, Docker Desktop/WSL2, PowerShell 7/Windows PowerShell, pytest, Cloudflare Quick Tunnel.

## Global Constraints

- Bind Gradio to `127.0.0.1:7860`. Direct sidecar runs bind `127.0.0.1:7862`; container runs bind `0.0.0.0:7862` only inside the container and MUST publish exactly `127.0.0.1:7862:7862`. Expose only Gradio through Cloudflare.
- Admit one request at a time. Run IMP first, nnU-Net second. No parallel GPU inference.
- Accept at most 16 MiB encoded input and 16,000,000 decoded pixels; normalize to contiguous RGB `uint8` exactly once.
- Arbitrary uploads show no Dice, BF1, IoU, HD95, ASSD, superiority, significance, protected-test, clinical, or SOTA claim.
- Fixed allowlisted train-screen samples retain their existing evidence class and provider-bound GT behavior.
- Pin nnU-Net model ID `L192-nnUNet-v2-raw-100ep`, checkpoint SHA-256 `3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2`, plans SHA-256 `b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7`, fingerprint SHA-256 `931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c`.
- Record Loop192 status `val_gate_failed_no_test`; never open protected test-v3 or imply promotion.
- Require CUDA for both public arms. No automatic CPU fallback. Abort launch on OOM, non-finite output, timeout, unexpected device, or artifact drift.
- Never use cached output to satisfy a live request. Sidecar failure clears the nnU-Net panel; incomplete runs cannot download a receipt.
- Receipts contain hashes, public model metadata, preprocessing, protocol, status, and latency only; no local path, username, environment variable, stack trace, or upload filename.
- Treat Cloudflare Quick Tunnel as temporary and unauthenticated. Use only synthetic or already-public rehearsal images.
- Never repair, unregister, reset, compact, resize, overwrite, import in place, or delete `Ubuntu-E` or `E:\WSL\Ubuntu-E\ext4.vhdx`.
- Attach the source VHD read-only and mount ext4 with `ro,noload`; abort unless both properties are proved before extraction.
- Preserve the existing forensic ivory/graphite/teal/rust design language; meet WCAG AA; support 1440x900 and 390x844; honor `prefers-reduced-motion`.

---

## File Structure

- `src/lesion_robustness/demo/dual_live_protocol.py`: canonical request/response schemas, RGB/mask hashing, strict validation, safe errors.
- `src/lesion_robustness/demo/nnunet_client.py`: loopback-only HTTP client, timeout, health check, response validation.
- `src/lesion_robustness/demo/dual_live_service.py`: sequential IMP then nnU-Net orchestration and stale-output clearing contract.
- `src/lesion_robustness/demo/presentation.py`: dual-live path-free receipt builder and public ledger rendering.
- `src/lesion_robustness/demo/app.py`: primary dual-live Gradio workflow; existing fixed-cache workflow moves to a secondary tab.
- `sidecar/nnunet/predictor.py`: persistent `nnUNetPredictor` adapter for one RGB image.
- `sidecar/nnunet/server.py`: localhost HTTP health/predict server with request limits and sanitized failures.
- `sidecar/nnunet/Dockerfile`: fresh pinned CUDA runtime; no dependency on damaged distro.
- `sidecar/nnunet/requirements.lock`: full reconstructed transitive pins generated inside the digest-pinned CUDA base; explicitly not the unavailable original environment.
- `sidecar/nnunet/model_manifest.example.json`: public schema plus immutable expected hashes; private artifact paths remain operator-local.
- `scripts/demo/verify_nnunet_bundle.py`: extraction verifier and path-free recovery receipt generator.
- `scripts/demo/recover_nnunet_artifacts.ps1`: administrator-only read-only VHD attachment/extraction/detachment.
- `scripts/demo/run_sidecar.ps1`: loopback-only Docker sidecar launcher and health probe.
- `scripts/demo/run_demo.ps1`: sidecar-first, CUDA dual-smoke preflight, guarded Gradio launch.
- `scripts/demo/stop_demo.ps1`: tunnel/host/sidecar shutdown checks and owned-temp cleanup.
- `tests/demo/test_dual_live_protocol.py`: protocol and hash contract tests.
- `tests/demo/test_nnunet_client.py`: loopback, timeout, malformed response, path-sanitization tests.
- `tests/demo/test_dual_live_service.py`: sequencing, live-only, partial-failure, receipt eligibility tests.
- `tests/demo/test_nnunet_sidecar.py`: predictor/server tests with a fake backend.
- `tests/demo/test_nnunet_recovery.py`: bundle hash and recovery-receipt tests.
- `tests/demo/test_app.py`: primary workflow, copy, queue, and output-clearing tests.
- `tests/demo/test_launch_scripts.py`: recovery/sidecar/preflight/tunnel/shutdown static and executable tests.
- `demo/README.md` and `docs/runbooks/demo-operations.md`: exact operator procedure and evidence limits.

---

### Task 1: Lock The Read-Only Recovery Contract

**Files:**
- Modify: `docs/superpowers/specs/2026-07-21-dual-live-demo-design.md`
- Create: `tests/demo/test_nnunet_recovery.py`
- Create: `scripts/demo/verify_nnunet_bundle.py`

**Interfaces:**
- Consumes: Loop192 report JSON and an extracted bundle directory.
- Produces: `verify_bundle(bundle: Path, report: Mapping[str, Any]) -> dict[str, Any]` and schema `loop192.recovery.receipt.v1`.

- [ ] **Step 1: Write failing bundle-verifier tests**

```python
def test_bundle_verifier_binds_required_hashes_and_omits_paths(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    receipt = verify_bundle(bundle, report)
    assert receipt["schema_version"] == "loop192.recovery.receipt.v1"
    assert receipt["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert receipt["plans_sha256"] == EXPECTED_PLANS_SHA256
    assert receipt["fingerprint_sha256"] == EXPECTED_FINGERPRINT_SHA256
    assert "path" not in json.dumps(receipt).lower()


def test_bundle_verifier_stops_on_hash_drift(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    (bundle / "checkpoint_final.pth").write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        verify_bundle(bundle, report)
```

- [ ] **Step 2: Run tests and verify the import failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_nnunet_recovery.py -v`

Expected: FAIL because `scripts.demo.verify_nnunet_bundle` does not exist.

- [ ] **Step 3: Implement strict bundle verification**

```python
REQUIRED = {
    "checkpoint_final.pth": "checkpoint_sha256",
    "nnUNetPlans.json": "plans_sha256",
    "dataset_fingerprint.json": "fingerprint_sha256",
}


def verify_bundle(bundle: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report["provenance"]
    observed: dict[str, str] = {}
    for filename, key in REQUIRED.items():
        value = sha256_file(bundle / filename)
        if value != str(provenance[key]):
            raise ValueError(f"{filename.split('.')[0]} hash mismatch")
        observed[key] = value
    extras = {}
    for filename in ("dataset.json", "plans.json", "runtime_identity.json", "requirements.lock"):
        path = bundle / filename
        if not path.is_file():
            raise FileNotFoundError(f"required Loop192 metadata missing: {filename}")
        extras[filename] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return {
        "schema_version": "loop192.recovery.receipt.v1",
        "model_id": str(report["candidate_id"]),
        **observed,
        "metadata": extras,
        "source_vhd_unchanged": True,
    }
```

- [ ] **Step 4: Re-run recovery tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_nnunet_recovery.py -v`

Expected: PASS; drift, missing file, malformed report, and path-leak cases pass.

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/specs/2026-07-21-dual-live-demo-design.md tests/demo/test_nnunet_recovery.py scripts/demo/verify_nnunet_bundle.py
git commit -m "test: lock read-only nnunet recovery"
```

### Task 2: Recover The Loop192 Bundle Without Writing The VHD

**Files:**
- Create: `scripts/demo/recover_nnunet_artifacts.ps1`
- Modify: `tests/demo/test_launch_scripts.py`
- Create operator-local, Git-ignored: `demo_runtime/nnunet/recovered/*`

**Interfaces:**
- Consumes: `-VhdPath`, `-ReportPath`, `-OutputRoot`, optional explicit `-PythonExe`, administrator token.
- Produces: verified artifact bundle plus `recovery_receipt.json`; returns `0` only after detach and unchanged source metadata checks.

- [ ] **Step 1: Add static safety tests**

```python
def test_recovery_script_is_read_only_and_forbids_distro_mutation() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")
    for token in ("-ReadOnly", "--bare", "ro,noload", "verify_nnunet_bundle.py"):
        assert token in script
    for forbidden in ("--unregister", "--import-in-place", "Resize-VHD", "Optimize-VHD"):
        assert forbidden not in script
    assert "Dismount-VHD" in script
    assert "finally" in script
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_launch_scripts.py::test_recovery_script_is_read_only_and_forbids_distro_mutation -v`

Expected: FAIL because the script is absent.

- [ ] **Step 3: Implement the guarded PowerShell workflow**

The script must: require elevation; verify the resolved VHD equals the explicit input; record length, creation time, last-write time; run `wsl --terminate Ubuntu-E`; use `Mount-VHD -Path $resolvedVhd -ReadOnly -Passthru`; prove `Get-Disk -Number $diskNumber` reports read-only; expose only that physical disk with `wsl --mount "\\.\PHYSICALDRIVE$diskNumber" --bare`; in `wsl --system`, locate the new ext4 block device, mount it with `mount -t ext4 -o ro,noload`; prove `/proc/mounts` contains both `ro` and `noload`; copy the exact allowlist; unmount; call `wsl --unmount`; call `Dismount-VHD` in `finally`; compare source length and timestamps; then call the verifier.

Artifact allowlist:

```powershell
$Required = @(
  'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_results/Dataset192_IMPlesionCleanV3RGB256/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_all/checkpoint_final.pth',
  'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_preprocessed/Dataset192_IMPlesionCleanV3RGB256/nnUNetPlans.json',
  'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_preprocessed/Dataset192_IMPlesionCleanV3RGB256/dataset_fingerprint.json',
  'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_raw/Dataset192_IMPlesionCleanV3RGB256/dataset.json',
  'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_results/Dataset192_IMPlesionCleanV3RGB256/nnUNetTrainer_100epochs__nnUNetPlans__2d/plans.json'
)
```

Also recover package identities from `.venv/lib/python3.12/site-packages/*.dist-info/METADATA`, `direct_url.json`, and `RECORD` for `nnunetv2`, `torch`, `dynamic-network-architectures`, `batchgenerators`, `batchgeneratorsv2`, `numpy`, `scipy`, `SimpleITK`, and `acvl-utils`. Generate `runtime_identity.json` plus a provisional recovery marker that states the original transitive lock is unavailable and reconstruction is required; never execute recovered code.

- [ ] **Step 4: Parse-check and dry-run the script**

Run: `powershell -NoProfile -Command "$errors=$null; [void][Management.Automation.Language.Parser]::ParseFile('scripts/demo/recover_nnunet_artifacts.ps1',[ref]$null,[ref]$errors); if($errors){$errors; exit 1}"`

Expected: exit `0`.

Run without elevation: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/recover_nnunet_artifacts.ps1 -PythonExe 'E:\0. IMP\.venv-win\Scripts\python.exe' -VhdPath 'E:\WSL\Ubuntu-E\ext4.vhdx' -ReportPath 'E:\0. IMP\.artifacts\preprocessing_search\current_bdou_loop192_nnunet_clean_v3_report.json' -OutputRoot 'E:\0. IMP\.worktrees\dual-live-demo\demo_runtime\nnunet\recovered'`

Expected: nonzero before any mount, message `Administrator token required`; no source metadata change.

- [ ] **Step 5: Run once elevated and inspect the receipt**

Run from an elevated PowerShell window:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/recover_nnunet_artifacts.ps1 `
  -PythonExe 'E:\0. IMP\.venv-win\Scripts\python.exe' `
  -VhdPath 'E:\WSL\Ubuntu-E\ext4.vhdx' `
  -ReportPath 'E:\0. IMP\.artifacts\preprocessing_search\current_bdou_loop192_nnunet_clean_v3_report.json' `
  -OutputRoot 'E:\0. IMP\.worktrees\dual-live-demo\demo_runtime\nnunet\recovered'
```

Expected: `recovery=passed`, the three pinned hashes match, source VHD length/timestamps unchanged, VHD detached.

- [ ] **Step 6: Commit code only**

```powershell
git add scripts/demo/recover_nnunet_artifacts.ps1 tests/demo/test_launch_scripts.py
git commit -m "feat: add read-only nnunet recovery"
```

### Task 3: Define The Hash-Bound Localhost Protocol

**Files:**
- Create: `src/lesion_robustness/demo/dual_live_protocol.py`
- Create: `tests/demo/test_dual_live_protocol.py`

**Interfaces:**
- Produces: `rgb_sha256(image)`, `mask_sha256(mask)`, `encode_request(request_id, image)`, `decode_request(payload)`, `decode_response(payload, expected)`, `SidecarResult`.
- Protocol ID: `imp.nnunet.sidecar.v1`.

- [ ] **Step 1: Write protocol tests**

```python
def test_request_round_trip_preserves_exact_rgb_and_digest() -> None:
    image = np.arange(12 * 9 * 3, dtype=np.uint8).reshape(12, 9, 3)
    payload = encode_request("a" * 32, image)
    request_id, decoded, digest = decode_request(payload)
    assert request_id == "a" * 32
    np.testing.assert_array_equal(decoded, image)
    assert digest == rgb_sha256(image)


@pytest.mark.parametrize("field", ["protocol", "request_id", "input_sha256", "mask_sha256", "checkpoint_sha256"])
def test_response_rejects_every_binding_drift(field: str) -> None:
    payload, expected = valid_response_fixture()
    payload[field] = "bad"
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_dual_live_protocol.py -v`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement canonical image and mask hashes**

```python
PROTOCOL_ID = "imp.nnunet.sidecar.v1"
MODEL_ID = "L192-nnUNet-v2-raw-100ep"


def rgb_sha256(image: np.ndarray) -> str:
    rgb = validate_rgb(image)
    prefix = f"{rgb.shape[0]}x{rgb.shape[1]}x3|".encode("ascii")
    return hashlib.sha256(prefix + rgb.tobytes(order="C")).hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    binary = validate_binary_mask(mask)
    prefix = f"{binary.shape[0]}x{binary.shape[1]}|".encode("ascii")
    return hashlib.sha256(prefix + binary.tobytes(order="C")).hexdigest()
```

Use lossless PNG base64 for transport; reject unknown fields, non-hex IDs/hashes, non-finite or negative latency, image/mask geometry drift, wrong model/checkpoint/protocol, response larger than 20 MiB, and decoded image above the global limits.

- [ ] **Step 4: Run protocol tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_dual_live_protocol.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/lesion_robustness/demo/dual_live_protocol.py tests/demo/test_dual_live_protocol.py
git commit -m "feat: define dual-live sidecar protocol"
```

### Task 4: Build The Persistent nnU-Net Sidecar

**Files:**
- Create: `sidecar/nnunet/predictor.py`
- Create: `sidecar/nnunet/server.py`
- Create: `sidecar/nnunet/Dockerfile`
- Create: `sidecar/nnunet/model_manifest.example.json`
- Create from the pinned base/runtime: `sidecar/nnunet/requirements.lock` (full reconstructed transitive lock, not the original environment)
- Create: `tests/demo/test_nnunet_sidecar.py`

**Interfaces:**
- Consumes: exact recovered model folder, model manifest, one decoded RGB image.
- Produces: `Loop192Predictor.predict(image: np.ndarray) -> tuple[np.ndarray, float]`; `GET /health`; `POST /v1/predict`.

- [ ] **Step 1: Write fake-backend server tests**

```python
def test_predict_endpoint_runs_backend_once_and_returns_bound_mask() -> None:
    backend = FakePredictor(mask=np.ones((13, 17), dtype=np.uint8))
    with running_sidecar(backend) as url:
        response = post_json(url + "/v1/predict", encode_request("a" * 32, rgb(13, 17)))
    assert backend.calls == 1
    decoded = decode_response(response, expected_bindings("a" * 32, rgb(13, 17)))
    assert decoded.execution == "live"
    assert decoded.mask.shape == (13, 17)


def test_sidecar_never_returns_path_or_trace_on_backend_error() -> None:
    with running_sidecar(BrokenPredictor("/private/checkpoint_final.pth")) as url:
        response = post_json(url + "/v1/predict", encode_request("b" * 32, rgb(8, 8)), expect=503)
    assert "private" not in json.dumps(response).lower()
    assert "traceback" not in json.dumps(response).lower()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_nnunet_sidecar.py -v`

Expected: FAIL because sidecar modules do not exist.

- [ ] **Step 3: Implement persistent predictor initialization**

The adapter verifies checkpoint/plans/fingerprint hashes before importing nnU-Net, requires `torch.cuda.is_available()`, sets `nnUNet_results` to the read-only model root, then initializes one predictor:

```python
self.predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device("cuda", 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False,
)
self.predictor.initialize_from_trained_model_folder(
    str(model_folder),
    use_folds=("all",),
    checkpoint_name="checkpoint_final.pth",
)
```

The recovered package identity test must assert the runtime's actual `nnUNetPredictor` method signatures before this code is accepted. Convert RGB to channel-first `float32`, attach the dataset JSON spacing contract, call `predict_single_npy_array`, require one finite binary label map, resize only the returned mask to original geometry with nearest-neighbor, synchronize CUDA around latency, then clear temporary tensors. No CLI subprocess and no per-request model reload.

- [ ] **Step 4: Implement the loopback server**

Default/direct execution uses `ThreadingHTTPServer(("127.0.0.1", 7862), Handler)`. Explicit container execution may use `0.0.0.0:7862` only inside the container namespace, paired with the exact host publication `127.0.0.1:7862:7862`; reject every other bind host. Guard prediction with one `Lock`. Set `Content-Length` limit before reading. Return only sanitized codes: `invalid_request`, `busy`, `oom`, `inference_failed`, `artifact_drift`. `/health` returns protocol, model ID, checkpoint hash, device, readiness, and no paths.

- [ ] **Step 5: Build the pinned container**

Use `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be`. Install `nnunetv2==2.8.1` inside that base and commit its full transitive `pip list --format=freeze` result as a clearly labeled reconstructed lock. Fail the build if the lock contains editable, local-path, VCS, or unpinned entries. Mount artifacts read-only at `/models/loop192`; publish `127.0.0.1:7862:7862`; request one GPU; set `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `nnUNet_compile=false`.

The original recorded output is unavailable. Replace that replay gate with the following complete Task 4 acceptance contract: exact artifact hashes; reconstructed dependency lock; CUDA/API identity; checkpoint load; same-current-runtime determinism on one public input; and live inference on two materially different public inputs with identical dimensions, distinct input hashes, and distinct current output hashes. Record input and output hashes without private paths. This does not prove original-runtime equivalence.

- [ ] **Step 6: Run unit tests and container identity probe**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_nnunet_sidecar.py -v`

Expected: PASS.

Run: `docker build -t imp-nnunet-sidecar:loop192 -f sidecar/nnunet/Dockerfile .`

Run: `docker run --rm --gpus all imp-nnunet-sidecar:loop192 python -c "import json,inspect,torch; from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor; print(json.dumps({'cuda':torch.cuda.is_available(),'init':str(inspect.signature(nnUNetPredictor.initialize_from_trained_model_folder)),'single':str(inspect.signature(nnUNetPredictor.predict_single_npy_array))}))"`

Expected: `cuda=true`; signatures match adapter calls.

Run the loaded checkpoint twice on one public input in the same current
container/runtime. Expected: identical current output hashes. Then run live
inference on two materially different public inputs with identical dimensions.
Expected: both requests succeed, their input hashes are distinct, and their
current output hashes are distinct. These checks establish current-runtime
determinism and input sensitivity only; they do not establish equivalence with
the unavailable original runtime.

- [ ] **Step 7: Commit**

```powershell
git add sidecar/nnunet tests/demo/test_nnunet_sidecar.py
git commit -m "feat: add pinned nnunet sidecar"
```

### Task 5: Add The Loopback Client And Sequential Orchestrator

**Files:**
- Create: `src/lesion_robustness/demo/nnunet_client.py`
- Create: `src/lesion_robustness/demo/dual_live_service.py`
- Create: `tests/demo/test_nnunet_client.py`
- Create: `tests/demo/test_dual_live_service.py`

**Interfaces:**
- `NnUNetClient(base_url="http://127.0.0.1:7862", timeout_seconds=90.0)`.
- `NnUNetClient.health() -> SidecarHealth`; `NnUNetClient.predict(request_id, image) -> SidecarResult`.
- `DualLiveService.run(image: np.ndarray) -> DualLiveResult`.

- [ ] **Step 1: Write client security tests**

```python
@pytest.mark.parametrize("url", ["http://0.0.0.0:7862", "http://localhost:7862", "https://127.0.0.1:7862", "http://127.0.0.1:9999"])
def test_client_accepts_only_exact_sidecar_origin(url: str) -> None:
    with pytest.raises(ValueError, match="exact loopback sidecar"):
        NnUNetClient(url)


@pytest.mark.parametrize(
    ("response", "failure"),
    [
        (TimeoutError("C:/private/model"), "timeout"),
        (FakeResponse(200, b"not-json"), "malformed_response"),
        (FakeResponse(200, json.dumps(response_with_wrong_mask_hash()).encode()), "binding_mismatch"),
    ],
)
def test_client_rejects_timeout_malformed_json_and_hash_drift(
    monkeypatch: pytest.MonkeyPatch, response: object, failure: str
) -> None:
    connection = FakeConnection(response)
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)
    client = NnUNetClient()
    with pytest.raises(SidecarUnavailable, match=failure) as raised:
        client.predict("a" * 32, rgb(8, 8))
    assert "private" not in str(raised.value).lower()
```

Define `FakeResponse.read(max_bytes)` and `FakeConnection.getresponse()` directly
in the test file; `FakeConnection.getresponse()` raises an exception value or
returns the configured response object.

- [ ] **Step 2: Write sequencing and fail-closed tests**

```python
def test_dual_service_runs_imp_then_nnunet_on_same_rgb() -> None:
    events: list[str] = []
    imp = FakeImp(events)
    nnunet = FakeNnUNet(events)
    image = rgb(24, 31)
    result = DualLiveService(imp, nnunet).run(image)
    assert events == ["imp", "nnunet"]
    assert result.input_sha256 == rgb_sha256(image)
    np.testing.assert_array_equal(imp.seen, nnunet.seen)
    assert result.receipt_eligible


def test_nnunet_failure_keeps_current_imp_but_forbids_receipt() -> None:
    result = DualLiveService(FakeImp([]), BrokenNnUNet()).run(rgb(8, 8))
    assert result.imp.status == "completed"
    assert result.nnunet.status == "failed"
    assert result.nnunet.mask is None and result.nnunet.overlay is None
    assert not result.receipt_eligible
```

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_nnunet_client.py tests/demo/test_dual_live_service.py -v`

Expected: FAIL on missing modules.

- [ ] **Step 4: Implement exact-origin HTTP client**

Use `http.client.HTTPConnection("127.0.0.1", 7862, timeout=90.0)`. Require status `200`, JSON content type, bounded response size, schema validation, request ID equality, input digest equality, pinned model/checkpoint, and mask digest equality. Convert socket, HTTP, JSON, and protocol failures to one sanitized `SidecarUnavailable(code)`.

- [ ] **Step 5: Implement sequential orchestration**

```python
def run(self, image: np.ndarray) -> DualLiveResult:
    rgb = validate_rgb(image).copy(order="C")
    request_id = secrets.token_hex(16)
    digest = rgb_sha256(rgb)
    imp = self.imp.preview_control(rgb.copy())
    try:
        nnunet = self.nnunet.predict(request_id, rgb.copy())
    except SidecarUnavailable as exc:
        return DualLiveResult.incomplete(request_id, digest, rgb, imp, exc.public_code)
    return DualLiveResult.complete(request_id, digest, rgb, imp, nnunet)
```

The result constructor validates both output geometries and binary masks. IMP failure returns no arm. nnU-Net failure retains only the current request's IMP result. No object stores a prior nnU-Net result.

- [ ] **Step 6: Run focused tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_nnunet_client.py tests/demo/test_dual_live_service.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/lesion_robustness/demo/nnunet_client.py src/lesion_robustness/demo/dual_live_service.py tests/demo/test_nnunet_client.py tests/demo/test_dual_live_service.py
git commit -m "feat: orchestrate sequential dual-live inference"
```

### Task 6: Build The Path-Free Dual-Live Receipt And Ledger

**Files:**
- Modify: `src/lesion_robustness/demo/presentation.py`
- Modify: `tests/demo/test_presentation.py`

**Interfaces:**
- Produces: `build_dual_live_receipt(result, registry) -> dict[str, Any]` and `render_dual_live_ledger(result) -> str`.

- [ ] **Step 1: Write receipt and evidence-scope tests**

```python
def test_dual_receipt_binds_input_outputs_models_protocol_and_latency() -> None:
    receipt = build_dual_live_receipt(complete_dual_result(), _registry())
    assert receipt["schema_version"] == "imp.dual_live.receipt.v1"
    assert receipt["execution"] == "live_sequential"
    assert receipt["evidence_class"] == "illustrative_arbitrary_upload_no_ground_truth"
    assert set(receipt["models"]) == {"imp", "nnunet"}
    assert receipt["models"]["nnunet"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    serialized = json.dumps(receipt).lower()
    for forbidden in ("dice", "iou", "bf1", "hd95", "assd", "path", "username", "diagnosis"):
        assert forbidden not in serialized


def test_incomplete_dual_result_cannot_build_receipt() -> None:
    with pytest.raises(ValueError, match="complete live result"):
        build_dual_live_receipt(incomplete_dual_result(), _registry())
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_presentation.py -v`

Expected: FAIL on missing builder.

- [ ] **Step 3: Implement canonical receipt and ledger**

Include: schema, request ID, `live_sequential`, evidence class, protocol ID, input RGB hash/dimensions, per-model ID/checkpoint/preprocessing/output-mask hash/latency/device/status, total latency, Loop192 `val_gate_failed_no_test`, and `clinical_use=false`. Sort receipt keys on disk; reject non-finite latency through JSON `allow_nan=False`.

- [ ] **Step 4: Run presentation tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_presentation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/lesion_robustness/demo/presentation.py tests/demo/test_presentation.py
git commit -m "feat: add dual-live evidence receipt"
```

### Task 7: Make Dual-Live The Primary Gradio Workflow

**Files:**
- Modify: `src/lesion_robustness/demo/app.py`
- Modify: `src/lesion_robustness/demo/theme.css`
- Modify: `tests/demo/test_app.py`

**Interfaces:**
- `create_app(service, registry, *, dual_service=None) -> gr.Blocks` preserves current callers.
- Primary callback returns original, IMP overlay/mask, nnU-Net overlay/mask, status, ledger, and receipt.

- [ ] **Step 1: Write app configuration and state-transition tests**

```python
def test_primary_tab_is_live_dual_model_compare() -> None:
    demo = create_app(FakeService(), _registry(), dual_service=FakeDualService())
    config = json.dumps(demo.get_config_file())
    assert config.index("Live Dual-Model Compare") < config.index("Exact Fixed-Cache Compare")
    assert "Run both models" in config
    assert "IMP mask" in config and "nnU-Net mask" in config
    assert demo._queue.default_concurrency_limit == 1


def test_incomplete_callback_clears_nnunet_and_disables_receipt() -> None:
    values = dual_component_values(incomplete_dual_result())
    assert values[3] is not None
    assert values[5] is None
    assert values[-1] is None
```

- [ ] **Step 2: Run app tests and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_app.py -v`

Expected: FAIL because dual-live controls do not exist.

- [ ] **Step 3: Implement one primary workflow**

Place upload/public sample selector and `Run both models` first. Render aligned columns: Original, IMP, nnU-Net. Keep exact fixed-cache comparison in `Audited Fixed Samples`; keep `Clean-v3 Evidence` and `Legacy Audit` secondary. Disable the run button when sidecar readiness is false. Use Gradio `concurrency_limit=1` and existing `concurrency_id="loop206-inference"`; keep `api_open=False`.

- [ ] **Step 4: Fix contrast and responsive design**

Set title foreground to graphite/rust on ivory, keep light text only on graphite surfaces, add `:focus-visible`, loading and failed state classes, 3-column desktop/1-column mobile layout, and a short staggered reveal. Verify every text/background pair using computed browser colors; do not rely on source CSS inspection alone.

- [ ] **Step 5: Run app and CSS tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_app.py tests/demo/test_presentation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/lesion_robustness/demo/app.py src/lesion_robustness/demo/theme.css tests/demo/test_app.py
git commit -m "feat: make dual-live comparison primary"
```

### Task 8: Launch Sidecar First And Gate Public Startup

**Files:**
- Create: `scripts/demo/run_sidecar.ps1`
- Create: `scripts/demo/stop_demo.ps1`
- Modify: `scripts/demo/run_demo.ps1`
- Modify: `scripts/demo/run_tunnel.ps1`
- Modify: `tests/demo/test_launch_scripts.py`

**Interfaces:**
- `run_sidecar.ps1 -CheckOnly` validates Docker, GPU, hashes, binding, and `/health`.
- `run_demo.ps1 -CheckOnly` requires sidecar health and performs one full dual smoke inference.
- `stop_demo.ps1` stops launcher-owned resources and proves ports 7860/7862 closed.

- [ ] **Step 1: Add launcher contract tests**

```python
def test_dual_launcher_gates_on_local_sidecar_and_cuda_smoke() -> None:
    demo = _read("scripts/demo/run_demo.ps1")
    sidecar = _read("scripts/demo/run_sidecar.ps1")
    assert "http://127.0.0.1:7862/health" in demo
    assert "dual_smoke=passed" in demo
    assert "--gpus" in sidecar and "127.0.0.1:7862:7862" in sidecar
    assert "0.0.0.0" not in sidecar
    assert "--network none" not in sidecar  # published loopback port requires bridge networking


def test_tunnel_exposes_only_gradio() -> None:
    script = _read("scripts/demo/run_tunnel.ps1")
    assert script.count("http://127.0.0.1:7860") >= 1
    assert "7862" not in script
```

- [ ] **Step 2: Run launcher tests and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_launch_scripts.py -v`

Expected: FAIL on missing sidecar launcher and dual preflight.

- [ ] **Step 3: Implement sidecar launcher**

Verify artifact receipt and Docker image identity before `docker run`. Use fixed container name `imp-nnunet-loop192`, `--rm`, `--gpus device=0`, `--publish 127.0.0.1:7862:7862`, read-only model mount, `--read-only`, `--tmpfs /tmp:rw,noexec,nosuid,size=256m`, memory limit, and restart policy `no`. Poll `/health` for at most 120 seconds; stop container on timeout.

- [ ] **Step 4: Extend Gradio launcher preflight**

Keep all existing Loop206 hash gates. Add exact sidecar health validation, then call an internal local-only dual-smoke function using one bundled public sample. Require CUDA device, both finite binary masks, both model IDs/checkpoint hashes, matching input hash, unique output hashes, and one complete path-free receipt. Stop before Gradio bind on failure.

- [ ] **Step 5: Implement ordered shutdown**

Stop Cloudflare first, then the Gradio process, then the named sidecar. Reuse the owned-session containment guard before deletion. Poll `Get-NetTCPConnection` until ports 7860 and 7862 are closed; exit nonzero if either remains.

- [ ] **Step 6: Run all launcher tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_launch_scripts.py -v`

Expected: PASS, including PowerShell parse tests and fake-process cleanup tests.

- [ ] **Step 7: Commit**

```powershell
git add scripts/demo/run_sidecar.ps1 scripts/demo/run_demo.ps1 scripts/demo/run_tunnel.ps1 scripts/demo/stop_demo.ps1 tests/demo/test_launch_scripts.py
git commit -m "feat: gate dual-live launch and shutdown"
```

### Task 9: Document Operation, Evidence Limits, And Two-Machine Transfer

**Files:**
- Modify: `demo/README.md`
- Modify: `docs/runbooks/demo-operations.md`
- Modify: `docs/runbooks/two-machine-delivery.md`
- Modify: `tests/demo/test_launch_scripts.py`

**Interfaces:**
- Produces one executable operator sequence for this machine and one RTX 4060 laptop setup sequence using GitHub plus local private artifact transfer.

- [ ] **Step 1: Add documentation contract tests**

```python
def test_runbook_documents_dual_live_limits_and_order() -> None:
    text = _read("docs/runbooks/demo-operations.md").lower()
    for token in ("run_sidecar.ps1", "run_demo.ps1", "run_tunnel.ps1", "stop_demo.ps1", "val_gate_failed_no_test", "no ground truth", "unauthenticated"):
        assert token in text
    assert text.index("sidecar") < text.index("gradio") < text.index("cloudflare")
```

- [ ] **Step 2: Run focused test and verify failure**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_launch_scripts.py::test_runbook_documents_dual_live_limits_and_order -v`

Expected: FAIL because runbook still describes control-only arbitrary uploads.

- [ ] **Step 3: Write exact operator procedures**

Document: recovery once; container build; local sidecar; guarded Gradio; local dual smoke; tunnel; outside-network smoke; tunnel stop; host stop; sidecar stop; port check. Mark arbitrary-upload output illustrative and unscored. State that source images may reach this host through an unauthenticated URL.

For the laptop: clone branch from GitHub; bootstrap Windows; install Docker Desktop GPU support and Cloudflared; transfer only the verified private bundle out-of-band; compare recovery receipt hashes; build sidecar; run `-CheckOnly`; never push weights, data, VHD, runtime receipts, or uploads to GitHub.

- [ ] **Step 4: Run documentation tests**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo/test_launch_scripts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add demo/README.md docs/runbooks/demo-operations.md docs/runbooks/two-machine-delivery.md tests/demo/test_launch_scripts.py
git commit -m "docs: add dual-live operating runbook"
```

### Task 10: Full Verification And Public Rehearsal

**Files:**
- Modify only if verification finds defects: files from Tasks 1-9.
- Create operator-local, Git-ignored: `demo_runtime/qa/*`.

**Interfaces:**
- Produces local test evidence, CUDA preflight receipt, desktop/mobile screenshots, one temporary Cloudflare URL smoke receipt, and closed-port proof.

- [ ] **Step 1: Run static and unit verification**

Run: `.venv-win\Scripts\python.exe -m pytest tests/demo -v`

Expected: all tests PASS.

Run: `.venv-win\Scripts\python.exe -m pytest -q`

Expected: full repository suite PASS; record count and elapsed time.

- [ ] **Step 2: Run hash and CUDA preflight**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -CheckOnly`

Expected: artifact hashes pass; Docker GPU visible; sidecar health identity matches Loop192.

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -CheckOnly`

Expected: existing Loop206 gates pass; `dual_smoke=passed`; no CPU fallback.

- [ ] **Step 3: Launch locally and verify real inference**

Start sidecar, then Gradio. Upload one already-public dermoscopy image. Confirm both output-mask hashes change when a materially different public image is used. Confirm the nnU-Net panel clears when the sidecar is stopped mid-session; no receipt download appears for that incomplete run. Restart sidecar before continuing.

- [ ] **Step 4: Perform browser visual QA**

At 1440x900 and 390x844 capture: initial, loading, successful dual result, sidecar unavailable, and invalid upload. Verify computed contrast, no clipping/horizontal scroll, aligned images, readable hashes, visible non-clinical warning, keyboard focus, and reduced-motion behavior. Fix defects, then repeat screenshots.

- [ ] **Step 5: Open a temporary tunnel and smoke from outside localhost**

Run the existing guarded tunnel only after local QA passes. From an external network, load the URL and run one synthetic/public sample. Confirm only port 7860 is public, sidecar metadata contains no path, metrics remain absent, and receipt hashes match the local result.

- [ ] **Step 6: Shut down and prove cleanup**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/stop_demo.ps1`

Expected: tunnel stopped first; ports 7860 and 7862 closed; named container absent; launcher-owned upload directory removed; unrelated runtime files preserved.

- [ ] **Step 7: Final repository audit**

Run: `git status --short`

Expected: only intentional tracked changes; no checkpoint, VHD, dataset, upload, temp receipt, screenshot containing public URL, or model bundle staged.

Run: `git log --oneline --decorate -12`

Expected: one reviewable commit per task.

- [ ] **Step 8: Commit any verification-only fixes**

```powershell
git add src/lesion_robustness/demo/dual_live_protocol.py src/lesion_robustness/demo/nnunet_client.py src/lesion_robustness/demo/dual_live_service.py src/lesion_robustness/demo/presentation.py src/lesion_robustness/demo/app.py src/lesion_robustness/demo/theme.css sidecar/nnunet scripts/demo tests/demo demo/README.md docs/runbooks/demo-operations.md docs/runbooks/two-machine-delivery.md
git commit -m "fix: close dual-live verification gaps"
```

Do not create this commit when no verification fix was needed.

---

## Scientific Acceptance Gate

The demo is ready for a defense only when all conditions hold:

1. Both models execute live on the same decoded RGB array and return current-request mask hashes.
2. Loop192 artifact hashes match the recorded report; runtime identity is pinned; protected test-v3 remains unopened.
3. Sequential CUDA smoke passes on this machine; RTX 4060 laptop repeats the same preflight before school use.
4. Arbitrary-upload UI and receipt contain no accuracy/comparative metric or superiority claim.
5. Fixed-sample metrics remain provider-authorized and visibly separated from arbitrary live inference.
6. Failure tests prove no cached nnU-Net output, stale panel, path leak, receipt on incomplete run, CPU fallback, or public sidecar binding.
7. Desktop/mobile visual QA passes after the hero contrast defect is fixed.
8. External smoke passes through a temporary Cloudflare URL; ordered shutdown closes both local ports.

**Evidence verdict if these pass:** Moderate demonstration evidence that two pinned implementations executed on submitted images. Still not clinical evidence, protected-test evidence, multi-seed model superiority, or SOTA evidence.
