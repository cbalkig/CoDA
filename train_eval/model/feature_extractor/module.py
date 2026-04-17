from typing import Optional, Iterator

import torch
from torch import nn
from torch.nn import Parameter

from configs.base.configs import Configs
from data.file.path import StoragePath
from data.model_spec import ModelSpec
from model.feature_extractor.model import FeatureExtractorModel
from model.managers.dropout_manager import DropoutType
from util.device_detector import DeviceDetector


class FeatureExtractor(nn.Module):
    def __init__(self, drop_rate: float, drop_path_rate: float) -> None:
        super().__init__()

        selected_model: ModelSpec = Configs().feature_extractor.timm_model
        pretrained: bool = Configs().feature_extractor.pretrained

        self._backbone = FeatureExtractorModel(
            selected_model=selected_model,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    @property
    def output_dim(self) -> int:
        return self._backbone.output_dim

    @property
    def number_of_layers(self) -> int:
        return self._backbone.number_of_layers

    @property
    def number_of_unfrozen_layers(self) -> int:
        return self._backbone.number_of_unfrozen_layers

    def freeze_all(self) -> None:
        self._backbone.freeze_all()

    def unfreeze_last_n_layers(self, n: Optional[int] = None) -> bool:
        if n is None:
            n = self.num_unfrozen_layers
            
        return self._backbone.unfreeze_last_n_layers(int(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def save(self, model_path: StoragePath) -> None:
        if model_path.local:
            model_path.path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": self.state_dict(),
            "unfrozen_layers": self.number_of_unfrozen_layers
        }, model_path.path)

    def load(self, model_path: StoragePath) -> None:
        device = DeviceDetector().device
        checkpoint = torch.load(model_path.path, map_location=device)
        self.load_state_dict(checkpoint["model_state"])
        self.freeze_all()
        self.unfreeze_last_n_layers(checkpoint["unfrozen_layers"])

    def build_layer_wise_lr_decay_param_groups(
            self, base_lr: float, gamma: Optional[float], weight_decay: float) -> Iterator[Parameter] | list[dict]:
        if gamma is None:
            return self.parameters()

        def _no_decay_name(n: str) -> bool:
            n = n.lower()
            return (
                    n.endswith(".bias")
                    or "norm" in n or "bn" in n or "layernorm" in n or "ln" in n
                    or "pos_embed" in n or "cls_token" in n
            )

        def _add_params_from_module(mod: nn.Module, lr: float):
            decay, no_decay = [], []
            for name, p in mod.named_parameters(recurse=True):
                if not p.requires_grad or id(p) in _seen:
                    continue
                _seen.add(id(p))
                if _no_decay_name(name):
                    no_decay.append(p)
                else:
                    decay.append(p)
            if decay:
                _groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                _groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

        # ---------- body ----------
        if not (base_lr > 0):
            raise ValueError("base_lr must be > 0")
        if not (0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")

        # Work on backbone if present; otherwise on self
        m = getattr(self._backbone, "_backbone", None)
        if m is None:
            m = getattr(self._backbone, "model", None)
        if m is None:
            m = self._backbone  # last resort

        _groups: list[dict] = []
        _seen: set[int] = set()

        # 2) Blocks: assign LR from TOP→DOWN with multiplicative decay
        top_lr = base_lr
        blocks, _tag = self._backbone.collect_blocks_bottom_to_top()
        for blk in reversed(blocks):  # reversed ==> top first
            _add_params_from_module(blk, base_lr)
            base_lr *= gamma  # decay as we go down

        # 3) Leftovers (e.g., embeddings/tokens or stray modules) at top LR
        #    Anything not yet seen gets base_lr (treated as "top-ish").
        extra_decay, extra_no_decay = [], []
        for name, p in m.named_parameters():
            if not p.requires_grad or id(p) in _seen:
                continue
            if _no_decay_name(name):
                extra_no_decay.append(p)
            else:
                extra_decay.append(p)

        leftover_lr = top_lr
        if extra_decay:
            _groups.append({"params": extra_decay, "lr": leftover_lr, "weight_decay": weight_decay})
        if extra_no_decay:
            _groups.append({"params": extra_no_decay, "lr": leftover_lr, "weight_decay": 0.0})

        if len(_groups) == 0:
            return self.parameters()

        return _groups

    def set_dropout(self, drop_rate: float, drop_path_rate: float) -> None:
        self._backbone.set_dropout(drop_rate, drop_path_rate)

    def get_dropout(self, dropout_type: DropoutType) -> float:
        return self._backbone.get_dropout(dropout_type)
