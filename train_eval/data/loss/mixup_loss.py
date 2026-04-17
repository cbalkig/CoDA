# add imports
from typing import Sequence, Optional

import torch
import torch.nn.functional as F
from timm.data import Mixup
from torch import Tensor

from data.loss.focal_loss import ClassBalancedFocalLoss
from util.device_detector import DeviceDetector


class MixupCriterion:
    def __init__(self, num_classes: int, class_weights: torch.Tensor, samples_per_cls: Sequence[int],
                 mixup_prob: float, switch_prob: Optional[float], mode: Optional[str], label_smoothing: Optional[float],
                 mixup_alpha: Optional[float], cutmix_alpha: Optional[float]) -> None:
        self._cb_focal = DeviceDetector().to(ClassBalancedFocalLoss(samples_per_cls))
        # derive normalized class weights from the CB focal (or recompute here)
        self.class_weights = DeviceDetector().to(class_weights)  # tensor [C]

        self.mixup_fn = None
        if mixup_prob > 0 and mixup_alpha is not None and cutmix_alpha is not None and (
                mixup_alpha > 0 or cutmix_alpha > 0):
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_minmax=None,
                prob=mixup_prob,
                switch_prob=switch_prob,
                mode=mode,  # try "elem" for stronger regularization
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )

        # if no mixup → keep your original CB focal
        self.use_soft_ce = self.mixup_fn is not None

    @torch.no_grad()
    def prepare_batch(self, x: Tensor, y: Tensor):
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)  # y becomes soft one-hot
        return x, y

    def _weighted_soft_ce(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: [B,C], targets: soft one-hot [B,C]
        logp = F.log_softmax(logits, dim=1)  # [B,C]
        loss = -(targets * logp)  # [B,C]
        loss = loss * self.class_weights.unsqueeze(0)  # apply class weights
        return loss.sum(dim=1).mean()

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self._weighted_soft_ce(logits, targets) if targets.ndim == 2 else self._cb_focal(logits, targets)
