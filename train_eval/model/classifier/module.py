from __future__ import annotations

import torch
from torch import nn

from configs.base.configs import Configs
from data.file.path import StoragePath
from model.classifier.model import MLPClassifierModel
from model.managers.dropout_manager import DropoutType
from util.device_detector import DeviceDetector


class MLPClassifier(nn.Module):
    """High-level façade around :class:`MLPClassifierModel` with save/load helpers."""

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            dropout: float,
    ) -> None:
        super().__init__()

        self._model = MLPClassifierModel(input_dim, Configs().classifier.hidden_dims, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def save(self, model_path: StoragePath) -> None:
        if model_path.local:
            model_path.path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), model_path.path)

    def load(self, model_path: StoragePath) -> None:
        device = DeviceDetector().device
        self.load_state_dict(torch.load(model_path.path, map_location=device))

    def set_dropout(self, p: float) -> None:
        self._model.set_dropout(p)

    def get_dropout(self, dropout_type: DropoutType) -> float:
        return self._model.get_dropout(dropout_type)
