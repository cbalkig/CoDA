from typing import Optional

import torch

from util.device_detector import DeviceDetector


class Loss:
    def __init__(self, value: Optional[torch.Tensor] = None, item_count: int = 0) -> None:
        self.value = (value if value is not None else DeviceDetector().to(torch.tensor(0.0)))
        self.item_count = item_count

    def add(self, loss: Optional["Loss"]) -> None:
        if loss is None:
            return
        self.value += loss.value
        self.item_count += loss.item_count

    def get_average(self) -> Optional[float]:
        if self.item_count == 0:
            return None
        return self.value.item() / self.item_count

    def detach(self) -> "Loss":
        return Loss(self.value.detach(), self.item_count)
