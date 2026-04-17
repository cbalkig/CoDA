import logging
import time
import traceback
from typing import List, Sequence, Tuple

import timm
import torch
from timm.layers import DropPath
from torch import nn, OutOfMemoryError

from data.model_spec import ModelSpec
from model.managers.dropout_manager import DropoutType
from util.device_detector import DeviceDetector

RETRY_SLEEP_SECS = 60
MAX_OOM_RETRIES = 10


class FeatureExtractorModel(nn.Module):
    def __init__(
            self,
            selected_model: ModelSpec,
            pretrained: bool,
            drop_rate: float,
            drop_path_rate: float,
    ) -> None:
        logging.getLogger("timm").setLevel(logging.WARNING)
        super().__init__()

        logging.warning(f'Selected model: {selected_model} - Pretrained: {pretrained}')

        self._model: nn.Module = DeviceDetector().to(timm.create_model(
            selected_model.model_name,
            features_only=True,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        ))

        self._keep_layernorm_fp32()

        self._dropout_modules: list[nn.Module] = []
        self._drop_path_modules: list[tuple[DropPath, float]] = []  # (module, rel_factor in [0,1])

        self._current_drop_rate = -1
        self._current_drop_path_rate = -1
        self._index_regularizers(drop_rate, drop_path_rate)

        logging.info(f"Feature extractor model initialized - Drop rate: {drop_rate}, Drop path: {drop_path_rate}")

    @property
    def output_dim(self) -> int:
        m = getattr(self._model, "model", self._model)
        if hasattr(m, "num_features"):
            return int(getattr(m, "num_features"))

        if hasattr(self._model, "feature_info"):
            try:
                chs = self._model.feature_info.channels()
                if chs and chs[-1] is not None:
                    return int(chs[-1])
            except Exception:
                pass

        # fallback: probe
        with torch.no_grad():
            device = next(self._model.parameters()).device
            x = torch.zeros(1, 3, 224, 224, device=device)
            ylist = self._model(x)
            y = ylist[-1] if isinstance(ylist, (list, tuple)) else ylist
            return int(y.shape[1])

    @property
    def number_of_layers(self) -> int:
        blocks = self.trainable_layers()
        return len(blocks) if blocks else 0

    @staticmethod
    def _has_params(mod: torch.nn.Module) -> bool:
        # True if the module has at least one parameter (regardless of requires_grad)
        return any(True for _ in mod.parameters(recurse=True))

    def trainable_layers(self) -> List[nn.Module]:
        """Blocks with at least one parameter (filters out Identity, Dropout-only, etc.)."""
        blocks, _ = self.layers
        return [b for b in blocks if self._has_params(b)]

    @property
    def number_of_unfrozen_layers(self) -> int:
        blocks = self.trainable_layers()
        if not blocks:
            return 0

        unfrozen = sum(
            any(p.requires_grad for p in b.parameters(recurse=True))
            for b in blocks
        )

        return unfrozen

    @property
    def layers(self) -> Tuple[Sequence[nn.Module], str]:
        m = getattr(self._model, "model", self._model)

        # ViT / DeiT
        if hasattr(m, "blocks") and isinstance(m.blocks, (list, tuple, nn.ModuleList, nn.Sequential)):
            return list(m.blocks), "vit"

        # ResNet-style
        if all(hasattr(m, f"layer{i}") for i in (1, 2, 3, 4)):
            blocks = []
            stem = []
            for name in ("conv1", "bn1", "act1", "relu", "maxpool"):
                if hasattr(m, name):
                    stem.append(getattr(m, name))
            if stem:
                blocks.append(nn.Sequential(*stem))
            for i in (1, 2, 3, 4):
                L = getattr(m, f"layer{i}")
                blocks.extend(list(L.children()))
            return blocks, "resnet"

        # RepVGG / VGGish via timm FeatureListNet (your screenshot)
        stage_names = [name for name, _ in m.named_children() if name.startswith("stages_")]
        if stage_names:
            stage_names = sorted(stage_names, key=lambda s: int(s.split("_")[1]))
            blocks = [getattr(m, "stem", None)] + [getattr(m, s) for s in stage_names]
            if hasattr(m, "final_conv"):
                blocks.append(getattr(m, "final_conv"))
            blocks = [b for b in blocks if b is not None]
            return blocks, "repvgg_stages"

        # RepVGG / ConvNeXt / RegNet
        if hasattr(m, "stages") and isinstance(m.stages, (list, tuple, nn.ModuleList, nn.Sequential)):
            blocks: List[nn.Module] = []
            for stage in m.stages:
                if isinstance(stage, nn.Sequential):
                    blocks.extend(list(stage))
                elif hasattr(stage, "blocks"):
                    blocks.extend(list(stage.blocks))
                else:
                    blocks.append(stage)
            return blocks, "stages"

        # Classic VGG: split .features at MaxPool2d boundaries into stages
        if hasattr(m, "features") and isinstance(m.features, torch.nn.Sequential):
            cur, blocks = [], []
            for mod in m.features:
                cur.append(mod)
                if isinstance(mod, torch.nn.MaxPool2d):
                    blocks.append(torch.nn.Sequential(*cur));
                    cur = []
            if cur: blocks.append(torch.nn.Sequential(*cur))
            return blocks, "vgg_features_grouped"

        return list(m.children()), "children"

    def _keep_layernorm_fp32(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.float()

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> bool:
        if n <= 0 or self.number_of_unfrozen_layers == n:
            return False

        logging.info(f"Feature extractor model unfreeze last layer: {n}")
        blocks = self.trainable_layers()
        if not blocks:
            logging.warning("Could not infer top blocks to unfreeze; leaving model frozen.")
            return False
        # freeze others
        for layer in blocks[:-n]:
            for p in layer.parameters():
                p.requires_grad = False

        # unfreeze last n
        blocks = [b for b in blocks if any(p.requires_grad or p.ndim >= 0 for p in b.parameters(recurse=True))]
        n = max(0, min(n, len(blocks)))
        for layer in blocks[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oom_retries = 0
        while True:
            try:
                feat_maps: List[torch.Tensor] = self._model(x)
                last_map = feat_maps[-1] if isinstance(feat_maps, (list, tuple)) else feat_maps
                return last_map.mean(dim=(-2, -1))
            except OutOfMemoryError as e:
                oom_retries += 1

                # Capture the current stack trace to identify the caller
                current_stack = "".join(traceback.format_stack())

                logging.warning(
                    f'Out of memory error (attempt {oom_retries}/{MAX_OOM_RETRIES}): {e}\n'
                    f'Caller Stack Trace:\n{current_stack}'
                )

                if oom_retries >= MAX_OOM_RETRIES:
                    raise
                time.sleep(RETRY_SLEEP_SECS)

    def _index_regularizers(self, drop_rate_init: float, drop_path_rate_init: float) -> None:
        m = getattr(self._model, "model", self._model)

        dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
        for mod in m.modules():
            if isinstance(mod, dropout_types):
                self._dropout_modules.append(mod)

        dp_modules: list[DropPath] = [mod for mod in m.modules() if isinstance(mod, DropPath)]
        if dp_modules:
            max_prob = max((dp.drop_prob for dp in dp_modules), default=0.0)
            if max_prob > 0:
                for dp in dp_modules:
                    rel = float(dp.drop_prob) / float(max_prob)  # in [0,1]
                    self._drop_path_modules.append((dp, rel))
            else:
                n = len(dp_modules)
                for i, dp in enumerate(dp_modules):
                    rel = 0.0 if n == 1 else i / (n - 1)
                    self._drop_path_modules.append((dp, rel))

        self._current_drop_rate = drop_rate_init
        self._current_drop_path_rate = drop_path_rate_init

    @torch.no_grad()
    def set_dropout(self, drop_rate: float, drop_path_rate: float) -> None:
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError(f"Feature Extractor - Drop rate must be in [0,1], current value is {drop_rate}.")

        if not (0.0 <= drop_path_rate <= 1.0):
            raise ValueError(f"Feature Extractor - Drop path rate must be in [0,1], current value is {drop_path_rate}.")

        self._current_drop_rate = 0.0 if drop_rate < 0.0 else (1.0 if drop_rate > 1.0 else float(drop_rate))
        self._current_drop_path_rate = 0.0 if drop_path_rate < 0.0 else (
            1.0 if drop_path_rate > 1.0 else float(drop_path_rate))

        for mod in self._dropout_modules:
            mod.p = self._current_drop_rate

        if self._drop_path_modules:
            rate = float(self._current_drop_path_rate)
            for dp, rel in self._drop_path_modules:
                if hasattr(dp, "drop_prob"):
                    dp.drop_prob = rate
                elif hasattr(dp, "p"):
                    dp.p = rate

        logging.info(
            f"Feature Extractor - Drop rate: {self._current_drop_rate}, Drop path: {self._current_drop_path_rate}")

    def get_dropout(self, dropout_type: DropoutType) -> float:
        if dropout_type == DropoutType.DROPOUT:
            return self._current_drop_rate

        if dropout_type == DropoutType.DROPOUT_PATH:
            return self._current_drop_path_rate

        raise NotImplementedError(f"Unknown dropout type: {dropout_type}")

    def collect_blocks_bottom_to_top(self) -> Tuple[List[nn.Module], str]:
        """
        Return backbone blocks ordered from BOTTOM (closest to input) to TOP (closest to classifier),
        plus a small string tag for the detected family.

        This simply wraps the `layers` helper, which already normalizes common backbones:
          - ViT/DeiT:     returns list(m.blocks)         -> [block0, ..., blockN-1]
          - ResNet:       stem + stages’ children        -> [stem, stage1_0, ..., stage4_k]
          - Staged nets:  flattens stages/sequentials
          - Features seq: list(m.features)
          - Fallback:     list(m.children())

        Notes:
        - Embeddings / tokens / pos_embed / cls_token / ln heads are NOT included here by design.
          Your LR builder will handle any “leftover” params by scanning the module afterward and
          assigning them the top LR.
        - Blocks with no trainable params are filtered out.
        """
        blocks, tag = self.layers  # uses your existing normalization logic
        # Filter empty blocks (no trainable params), keep ordering intact
        filtered: List[nn.Module] = []
        for b in blocks:
            try:
                has_params = any(p.requires_grad for p in b.parameters(recurse=True))
            except Exception:
                has_params = False
            if has_params:
                filtered.append(b)
        return filtered, tag
