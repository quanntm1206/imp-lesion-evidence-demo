# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    nnUNet_compile=false \
    NNUNET_BIND_HOST=0.0.0.0 \
    NNUNET_MODEL_ROOT=/models/loop192 \
    NNUNET_MODEL_MANIFEST=/app/sidecar/nnunet/model_manifest.example.json \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY sidecar/nnunet/requirements.lock /tmp/requirements.lock
RUN python - <<'PY'
from pathlib import Path
import re

lock = Path("/tmp/requirements.lock").read_text(encoding="utf-8").splitlines()
pins = [line for line in lock if line and not line.startswith("#")]
for line in pins:
    lowered = line.lower()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+", line):
        raise SystemExit("requirements.lock contains an unpinned entry")
    if any(token in lowered for token in ("git+", "http://", "https://", "file:", " @ ", "-e ")):
        raise SystemExit("requirements.lock contains a forbidden source")
if "nnunetv2==2.8.1" not in pins:
    raise SystemExit("requirements.lock does not pin nnunetv2==2.8.1")
PY
RUN python -m pip install --break-system-packages --no-cache-dir \
        --requirement /tmp/requirements.lock \
    && python -m pip check \
    && rm /tmp/requirements.lock

COPY src/lesion_robustness/demo/dual_live_protocol.py /app/src/lesion_robustness/demo/dual_live_protocol.py
COPY sidecar/nnunet /app/sidecar/nnunet

RUN groupadd --gid 65532 sidecar \
    && useradd --uid 65532 --gid 65532 --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin sidecar
USER sidecar:sidecar
CMD ["python", "-m", "sidecar.nnunet.server"]
