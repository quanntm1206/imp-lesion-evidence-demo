# Robust Skin Lesion Segmentation under Image Quality Variation

MVP research scaffold for:

> Cải thiện độ bền vững của phân đoạn tổn thương da dưới điều kiện ảnh thay đổi chất lượng bằng xử lý ảnh cổ điển và deep learning.

## Scope

- Main dataset: ISIC 2018 Task 1-style image/mask folders.
- Core baselines:
  - classical-only: CLAHE + Otsu threshold + morphology + largest component.
  - U-Net baseline.
  - hybrid: CLAHE/filtering preprocessing + U-Net + morphology post-processing.
- Robustness corruptions:
  - low brightness
  - low contrast
  - Gaussian noise
  - Gaussian blur
  - JPEG compression
  - optional uneven illumination / hair-like artifact from code.
- Metrics: Dice, IoU, precision, recall.

## Tracked release scope

This rescue branch packages two tracked command-line entry points:

- `lesion-demo` for the guarded evidence-first workbench.
- `lesion-build-evidence` for the canonical evidence registry.

Historical training, classical-baseline, evaluation, threshold-tuning, and Linux helper modules remain outside this tracked release. Their old commands are intentionally not advertised as clone-runnable entry points.

## Notes

- Keep test-set corruption generation deterministic by passing a fixed seed where applicable.
- Tune threshold/kernel/min-area on validation only, not on test.
- U-Net++ and Attention U-Net are intended next baselines; this scaffold starts with a lightweight U-Net to keep the first runnable version small.

## Paper/demo rescue delivery

The evidence-bound paper and non-clinical demo use a private two-machine workflow. On a clean Windows clone, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
```

This rescue check is separate from the Linux training workflow. It creates `.venv-win`, runs portable CPU demo tests, rebuilds tracked paper evidence, records the registry-only audit scope, then compiles the paper. Optional CUDA 13.0 environment verification uses `-Compute cu130`; model weights remain off GitHub. Strict source-byte release audit remains a main-workstation gate.

Browser rendering and desktop/mobile screenshots remain unverified. The runbooks describe required operator checks; they are not recorded release evidence.

See [`docs/runbooks/two-machine-delivery.md`](docs/runbooks/two-machine-delivery.md) for machine roles, private branch policy, clean-clone handoff, CI receipts, and hash-verified artifact transfer.
