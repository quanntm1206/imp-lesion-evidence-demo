"""Test-only nnU-Net client decorator that times out exactly once."""

from __future__ import annotations

from typing import Protocol
import threading

import numpy as np

from lesion_robustness.demo.dual_live_protocol import SidecarResult
from lesion_robustness.demo.nnunet_client import SidecarUnavailable


class _Predictor(Protocol):
    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult: ...


class OneShotTimeoutClient:
    def __init__(self, client: _Predictor) -> None:
        self._client = client
        self._pending_timeout = True
        self._state_lock = threading.Lock()

    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult:
        with self._state_lock:
            timeout = self._pending_timeout
            self._pending_timeout = False
        if timeout:
            raise SidecarUnavailable("timeout")
        return self._client.predict(request_id, image)
