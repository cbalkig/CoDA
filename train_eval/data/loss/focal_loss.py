import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base.configs import Configs
from util.device_detector import DeviceDetector


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_classes):
        super().__init__()
        self.reduction = Configs().focal_loss.reduction
        self.gamma = float(Configs().focal_loss.gamma)

        # Class-balanced alpha (inverse effective number, normalized to mean=1)
        beta = float(Configs().focal_loss.beta)
        eff_num = torch.tensor([(1.0 - beta ** n) / (1.0 - beta) for n in samples_per_classes],
                               dtype=torch.float32)
        inv = 1.0 / eff_num
        alpha = inv / inv.mean()
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = DeviceDetector().to(self.alpha)

        if targets.ndim == 2:
            # Soft-label path (mixup on)
            logp = F.log_softmax(logits, dim=1)
            loss = -(targets * logp) * alpha.unsqueeze(0)
            loss = loss.sum(dim=1)
        else:
            # Hard-label path (no mixup)
            logpt = F.log_softmax(logits, dim=1)
            pt = logpt.exp()
            ce = F.nll_loss(logpt, targets, reduction="none")
            focal_factor = (1.0 - pt.gather(1, targets.view(-1, 1)).squeeze(1)) ** self.gamma
            loss = alpha[targets] * focal_factor * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
