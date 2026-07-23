# Hướng Dẫn Tự Triển Khai Demo Cho Giảng Viên

## 1. Phạm vi

Đây là demo nghiên cứu phi lâm sàng. Kết quả chỉ là ảnh minh họa phân vùng,
không phải chẩn đoán và không chứng minh model nào vượt trội.

**Không thể chạy inference thật chỉ từ GitHub.** Repository không chứa
checkpoint, dataset, fixed cache, evidence report hoặc reconstructed nnU-Net
bundle. Giảng viên cần nhận thêm gói private artifact từ nhóm qua USB hoặc kênh
riêng, kèm `sha256-manifest.json`. Không nhận đủ gói này thì dừng ở bước kiểm
tra source; không thay bằng weights giả hoặc model khác.

Demo chạy tuần tự trên cùng một ảnh RGB:

1. IMP MiT-B3 U-Net (`L206-control-s206`).
2. Reconstructed nnU-Net (`L192-nnUNet-v2-raw-100ep`).

## 2. Máy yêu cầu

- Windows 10/11 64-bit.
- NVIDIA GPU hỗ trợ CUDA; 8 GB VRAM là cấu hình mục tiêu, nhưng launcher mới là
  kiểm tra quyết định.
- NVIDIA driver, Docker Desktop với WSL2 và GPU support.
- Git, GitHub CLI, `uv`, Python 3.12.
- `cloudflared` chỉ cần khi muốn mở link tạm thời ra Internet.
- Quyền truy cập private repository
  `quanntm1206/imp-lesion-evidence-demo`.

Kiểm tra công cụ trong PowerShell:

```powershell
git --version
gh auth status
uv --version
docker version
nvidia-smi
cloudflared --version
```

Nếu không dùng Cloudflare, lỗi ở lệnh `cloudflared --version` có thể bỏ qua.

## 3. Clone đúng branch

```powershell
gh repo view quanntm1206/imp-lesion-evidence-demo --json isPrivate,url
gh repo clone quanntm1206/imp-lesion-evidence-demo
Set-Location imp-lesion-evidence-demo
git switch codex/dual-live-demo
git rev-parse HEAD
```

Kết quả `isPrivate` phải là `true`. Ghi lại commit SHA để báo lại nhóm nếu có
lỗi.

## 4. Cài môi trường demo

Các lệnh sau cài dependency rồi phủ đúng CUDA wheels. Không chạy `uv sync`
lại sau bước CUDA overlay vì nó có thể thay torch GPU bằng bản CPU.

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-win'
uv sync --locked --python 3.12 --extra dev --extra analysis --extra demo --extra train
uv pip install --python .venv-win\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu130/ torch==2.12.0+cu130 torchvision==0.27.0+cu130
$PythonExe = (Resolve-Path '.venv-win\Scripts\python.exe').Path
& $PythonExe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

Nếu máy không tương thích CUDA 13.0, không tự đổi phiên bản trong launcher.
Gửi commit SHA, GPU, driver và lỗi cho nhóm để tạo một runtime đã được review.

## 5. Gói private artifact bắt buộc

Giải nén phần `repository-overlay` vào thư mục clone và giữ nguyên cấu trúc.
Gói bàn giao tối thiểu phải tạo ra các đường dẫn sau:

Hai entry bắt buộc là `demo_runtime/loop206_dataset_index.json` và bundle
`demo_runtime/nnunet/recovered-container-final2`.

```text
.artifacts/preprocessing_search/
  current_bdou_loop191_raw_rater_uncertainty_report.json
  current_bdou_loop192_nnunet_clean_v3_report.json
  current_bdou_loop206_final_closure_report.json
  loop206_leac_drlse/pilot_cache_v2_candidate/manifest.json
  loop206_leac_drlse/pilot_cache_v2_candidate/<data file declared by manifest>
  loop206_leac_drlse/pilot_cache_v2_zero_control/manifest.json
  loop206_leac_drlse/pilot_cache_v2_zero_control/<data file declared by manifest>
demo_runtime/
  loop206_dataset_index.json
  nnunet/recovered-container-final2/
    checkpoint_final.pth
    dataset.json
    dataset_fingerprint.json
    plans.json
    recovery_receipt.json
```

Gói private còn phải có:

- IMP control checkpoint.
- IMP candidate checkpoint; cần cho audited fixed-sample surface và preflight.
- Root chứa ảnh công khai được dataset index tham chiếu.
- Docker image archive đã xác minh, hoặc đủ điều kiện build đúng pinned image.
- `sha256-manifest.json` chứa relative path, byte size và SHA-256 của mọi file.

Không commit hoặc upload các file trên lên GitHub.

### Kiểm tra manifest bàn giao

Đặt biến `IMP_DEMO_ARTIFACT_ROOT` tới thư mục gốc của gói nhận được:

```powershell
$ArtifactRoot = (Resolve-Path -LiteralPath $env:IMP_DEMO_ARTIFACT_ROOT).Path
$ManifestPath = Join-Path $ArtifactRoot 'sha256-manifest.json'
$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
foreach ($Entry in $Manifest.files) {
  $Relative = [string]$Entry.path
  if ([IO.Path]::IsPathRooted($Relative) -or $Relative -match '(^|/)\.\.(/|$)') { throw "Unsafe artifact path: $Relative" }
  $File = Join-Path $ArtifactRoot $Relative.Replace('/', '\')
  if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing artifact: $Relative" }
  if ((Get-Item -LiteralPath $File).Length -ne [int64]$Entry.bytes) { throw "Size mismatch: $Relative" }
  if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant() -ne ([string]$Entry.sha256).ToLowerInvariant()) { throw "SHA-256 mismatch: $Relative" }
}
'artifact_manifest=passed'
```

Sau đó trỏ ba biến môi trường tới file/root private đã kiểm tra:

```powershell
$env:IMP_LOOP206_CONTROL_CHECKPOINT = (Resolve-Path -LiteralPath (Join-Path $ArtifactRoot 'private/imp_control.pt')).Path
$env:IMP_LOOP206_CANDIDATE_CHECKPOINT = (Resolve-Path -LiteralPath (Join-Path $ArtifactRoot 'private/imp_candidate.pt')).Path
$env:IMP_LOOP206_DATA_ROOT = (Resolve-Path -LiteralPath (Join-Path $ArtifactRoot 'private/data-root')).Path
```

Nếu gói dùng tên khác, chọn đúng entry trong `sha256-manifest.json`; không đổi
tên model, digest hoặc nội dung registry.

## 6. Chuẩn bị reconstructed nnU-Net image

Launcher yêu cầu tag `imp-nnunet-sidecar:loop192` có đúng image ID:

```text
sha256:86bd77c03c3918e3638565e29417cdf4360b499a0813fbc425dc36645f026f2d
```

Nếu gói private có Docker archive đã hash-verified:

```powershell
docker load --input (Join-Path $ArtifactRoot 'private/imp-nnunet-sidecar-loop192.tar')
docker image inspect --format '{{.Id}}' imp-nnunet-sidecar:loop192
```

Nếu tự build:

```powershell
docker build -t imp-nnunet-sidecar:loop192 -f sidecar/nnunet/Dockerfile .
docker image inspect --format '{{.Id}}' imp-nnunet-sidecar:loop192
```

Nếu ID khác, dừng. Không sửa pin trong `run_sidecar.ps1`; yêu cầu nhóm cung
cấp image archive đúng hash.

## 7. Chạy demo local

Mở ba cửa sổ PowerShell tại repository. Tạo một Run ID, rồi chép đúng giá trị
đó sang cả ba cửa sổ:

```powershell
$RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffffffZ').ToLowerInvariant()
$PythonExe = (Resolve-Path '.venv-win\Scripts\python.exe').Path
$RunId
```

### Terminal 1 - nnU-Net sidecar

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -CheckOnly -PreserveMode -RunId $RunId
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_sidecar.ps1 -PreserveMode -RunId $RunId
```

Chỉ tiếp tục khi thấy `sidecar_health=passed`. Sidecar chỉ được bind tại
`127.0.0.1:7862`; không mở port này trên router hoặc firewall.

### Terminal 2 - Gradio và dual-model smoke

Đặt lại cùng `$RunId`, `$PythonExe` và ba biến `IMP_LOOP206_*`, sau đó chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -CheckOnly -PublicTunnelMode -PreserveMode -RunId $RunId -PythonExe $PythonExe
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_demo.ps1 -PublicTunnelMode -PreserveMode -RunId $RunId -PythonExe $PythonExe
```

Preflight phải in `preflight=passed` và `dual_smoke=passed`. Mở
`http://127.0.0.1:7860`, chọn một bundled public sample, rồi xác nhận đủ ba
panel Original, IMP và nnU-Net. Public mode không mở arbitrary-upload API.

## 8. Mở link tạm thời bằng Cloudflare (tùy chọn)

Chỉ làm bước này sau khi demo local chạy thành công. Terminal 3 phải dùng cùng
`$RunId`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/run_tunnel.ps1 -PreserveMode -RunId $RunId
```

Launcher chỉ expose Gradio tại `127.0.0.1:7860`; sidecar
`127.0.0.1:7862` vẫn local. Dùng URL tạm thời mới được in ra terminal. Không
ghi URL vào repository, slide, ảnh chụp hoặc email công khai. Chỉ dùng bundled
public/synthetic samples; tunnel không có authentication.

## 9. Checklist nghiệm thu

- Sidecar: `sidecar_health=passed`.
- Demo preflight: `preflight=passed` và `dual_smoke=passed`.
- Local browser: Original, IMP, nnU-Net đều thuộc request hiện tại.
- Chạy hai bundled public samples khác nhau; output hashes phải khác nhau.
- Lặp lại một sample chỉ để kiểm tra determinism của runtime hiện tại.
- Không diễn giải mask như diagnosis, accuracy hoặc clinical result.
- Nếu dùng Cloudflare, kiểm tra từ mạng khác; chỉ port 7860 được public.

## 10. Dừng đúng thứ tự

Trong một PowerShell tại repository, đặt lại cùng `$RunId`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo/stop_demo.ps1 -PreserveMode -RunId $RunId
```

Script dừng Cloudflare, Gradio, rồi sidecar. Xác nhận không còn listener:

```powershell
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -In 7860,7861,7862
```

Kết quả phải rỗng. Không chỉ đóng terminal hoặc dùng `Ctrl+C`.

## 11. Troubleshooting

| Lỗi | Nguyên nhân thường gặp | Cách xử lý an toàn |
|---|---|---|
| `release artifact hash` | Thiếu/sai dataset index, cache manifest, cache data hoặc checkpoint | Kiểm lại `sha256-manifest.json`; không sửa digest trong code |
| `evidence source hash` | Thiếu ba private evidence reports tại `.artifacts/preprocessing_search` | Chép lại đúng repository overlay từ gói bàn giao |
| `pinned image probe` | Docker image ID không đúng | Dùng image archive đã xác minh; không đổi pin |
| `CUDA unavailable` | Driver, Docker GPU hoặc torch CUDA chưa hoạt động | Chạy lại GPU checks; gửi log đã scrub cho nhóm |
| `sidecar health` | Bundle nnU-Net thiếu file hoặc recovery receipt sai | Kiểm đủ năm file và SHA-256 trong manifest |
| Port 7860/7862 đã dùng | Runtime cũ chưa dừng | Chạy `stop_demo.ps1` với đúng Run ID; không kill process không rõ ownership |
| Cloudflare `1033` | Tunnel cũ đã tắt hoặc URL đã hết hạn | Chạy tunnel mới sau local preflight; dùng URL mới được in ra |
| Chỉ có source GitHub | Không có private runtime | Không thể demo inference thật; liên hệ nhóm để nhận artifact bundle |

Khi báo lỗi, gửi commit SHA, Run ID, bước thất bại và dòng lỗi đã loại bỏ path
private. Không gửi checkpoint, dataset, receipt, absolute path hoặc tunnel URL.
