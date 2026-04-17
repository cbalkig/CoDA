import logging
import os
import time
import traceback
from typing import Tuple, Dict

from torch import OutOfMemoryError

from configs.base.configs import Configs

# Ensure allocator config is in place *before* any CUDA context is created
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn


class DeviceDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceDetector, cls).__new__(cls)
            cls._instance._device, cls._instance._device_count, cls._instance._device_names = cls._detect_device()
        return cls._instance

    @staticmethod
    def _detect_device() -> Tuple[torch.device, int, Dict[int, str]]:
        logging.warning(f"Torch: {torch.__version__}")
        logging.warning(f"Torch CUDA version: {torch.version.cuda}")
        logging.warning(f"Torch CUDNN: {torch.backends.cudnn.version()}")

        device_names: Dict[int, str] = {}

        # ---------- CPU (forced) ----------
        if Configs().training.force_cpu:
            logging.warning("Force CPU mode")
            device = torch.device("cpu")
            device_count = os.cpu_count() or 1  # logical CPU cores
            device_names[0] = "CPU"

        # ---------- Apple M‑series / MPS ----------
        elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
        ):
            logging.warning("MPS mode")
            device = torch.device("mps")
            # guard: torch.mps.device_count() exists only in recent PyTorch versions
            device_count = getattr(torch.mps, "device_count", lambda: 1)()
            device_names[0] = "MPS"

        # ---------- CUDA ----------
        elif torch.cuda.is_available():
            logging.warning("CUDA mode")

            try:
                gpu_id = int(Configs().training.gpu_id)  # may be "0", "1", None, etc.
            except (AttributeError, ValueError, TypeError):
                logging.warning("ConfigUtil().gpu_id is not set or not an int ‑> defaulting to 0")
                gpu_id = 0

            device_count = torch.cuda.device_count()
            if gpu_id < 0 or gpu_id >= device_count:
                logging.warning(
                    f"Requested GPU id {gpu_id} out of range (0‑{device_count - 1}) "
                    "‑> defaulting to 0"
                )
                gpu_id = 0

            selected_gpu = f"cuda:{gpu_id}"
            device = torch.device(selected_gpu)
            torch.cuda.set_device(device)

            logging.warning(
                f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(device)}"
            )

            for idx in range(device_count):
                device_names[idx] = torch.cuda.get_device_name(idx)

        # ---------- No accelerator detected ----------
        else:
            logging.warning("NONE, Moving to CPU mode")
            device = torch.device("cpu")
            device_count = os.cpu_count() or 1
            device_names[0] = "CPU"

        logging.warning(f"DEVICE: {device}")
        logging.warning(f"DEVICE COUNT: {device_count}")
        logging.warning(f"DEVICE NAMES: {device_names}")
        return device, device_count, device_names

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_count(self) -> int:
        return self._device_count

    @property
    def device_names(self) -> Dict[int, str]:
        return self._device_names

    @property
    def cpu(self) -> torch.device:
        """Convenience accessor for a CPU device."""
        return torch.device("cpu")

    def empty_cache(self) -> None:
        """Release unused GPU / MPS memory."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()

    def to(self, x: torch.nn.Module | torch.Tensor) -> torch.nn.Module | torch.Tensor:
        try:
            return x.to(self.device)
        except OutOfMemoryError as e:
            # Capture the current stack trace to identify the caller
            current_stack = "".join(traceback.format_stack())

            logging.warning(
                f'Out of memory error: {e}\n'
                f'Caller Stack Trace:\n{current_stack}'
            )

            DeviceDetector().empty_cache()
            time.sleep(60)
            return self.to(x)
