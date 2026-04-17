from __future__ import annotations

import logging

import torch
from torch import nn

from model.managers.dropout_manager import DropoutType


class MLPClassifierModel(nn.Module):
    """
    Plain MLP classifier:
      - Linear (no bias) + BatchNorm1d + ReLU (+ optional Dropout) repeated for hidden layers
      - Final Linear to output_dim (bias=True)
    Notes:
      * No device placement or .eval() in __init__. The caller controls .to(device) and .train()/.eval().
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            output_dim: int,
            dropout_rate: float,
    ) -> None:
        super().__init__()

        self._current_drop_rate = 0.0 if dropout_rate < 0.0 else (1.0 if dropout_rate > 1.0 else float(dropout_rate))
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h, bias=False))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(float(self._current_drop_rate)))
            prev_dim = h

        # Final classifier layer (keep bias=True)
        layers.append(nn.Linear(prev_dim, output_dim))

        # No implicit device move or mode change here
        self.net = nn.Sequential(*layers)

        self._init_weights()

        logging.info(f"Classifier model initialized - Dropout rate: {dropout_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim)
        return self.net(x)  # (B, output_dim)

    def _init_weights(self) -> None:
        """Xavier-uniform for Linear layers; zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def set_dropout(self, dropout_rate: float) -> None:
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"Classifier dropout rate must be in [0,1], got {dropout_rate}")

        self._current_drop_rate = 0.0 if dropout_rate < 0.0 else (1.0 if dropout_rate > 1.0 else float(dropout_rate))

        DROPOUT_TYPES = (
            nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
            nn.AlphaDropout, nn.FeatureAlphaDropout
        )

        seen = set()

        rate = float(self._current_drop_rate)
        for m in self.modules():
            # direct modules
            if isinstance(m, DROPOUT_TYPES) and id(m) not in seen:
                m.p = rate
                seen.add(id(m))

            # common attribute names that point to submodules
            for attr in ("dropout", "drop"):
                d = getattr(m, attr, None)
                if isinstance(d, DROPOUT_TYPES) and id(d) not in seen:
                    d.p = rate
                    seen.add(id(d))

        logging.info(
            f"Classifier - Drop rate: {self._current_drop_rate}")

    def get_dropout(self, dropout_type: DropoutType) -> float:
        if dropout_type == DropoutType.DROPOUT:
            return self._current_drop_rate

        raise NotImplementedError(f"dropout type {dropout_type} not implemented")
