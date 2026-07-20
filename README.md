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

## Linux VM usage

From the repository root inside the Ubuntu WSL VM:

```bash
bash scripts/setup_linux.sh
bash scripts/run_tests.sh
```

The Windows host is only used as the mounted workspace; training/evaluation commands should run in Linux.

## Expected data layout

```text
data/isic2018-task1/
  images/
    ISIC_0000000.jpg
  masks/
    ISIC_0000000_segmentation.png
```

If your downloaded folders have different names, update `configs/default.yaml`.

## Run classical baseline

```bash
uv run --python 3.12 lesion-classical \
  --images-dir data/isic2018-task1/images \
  --masks-dir data/isic2018-task1/masks \
  --image-size 256
```

## Train U-Net

Install the training extra first after confirming CUDA is visible in WSL:

```bash
uv sync --python 3.12 --extra dev --extra analysis --extra train
uv run --python 3.12 --extra train lesion-train --config configs/default.yaml
```

Best checkpoint is saved to:

```text
runs/checkpoints/best.pt
```

## Evaluate clean/corrupted robustness

```bash
uv run --python 3.12 --extra train lesion-evaluate \
  --config configs/default.yaml \
  --checkpoint runs/checkpoints/best.pt \
  --split test \
  --corruption clean

uv run --python 3.12 --extra train lesion-evaluate \
  --config configs/default.yaml \
  --checkpoint runs/checkpoints/best.pt \
  --split test \
  --corruption low_contrast
```

Use the drop:

```text
Robustness Drop = Dice_clean - Dice_corrupted
```

## Recommended experiments

1. Classical-only on clean test.
2. U-Net on clean test.
3. U-Net on corrupted test sets.
4. Hybrid CLAHE/filter + morphology post-processing.
5. Ablation:
   - no preprocessing/no post-processing
   - CLAHE only
   - median filter only
   - morphology only
   - CLAHE + median + morphology

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

See [`docs/runbooks/two-machine-delivery.md`](docs/runbooks/two-machine-delivery.md) for machine roles, private branch policy, clean-clone handoff, CI receipts, and hash-verified artifact transfer.
