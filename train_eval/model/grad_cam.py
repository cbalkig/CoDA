#!/usr/bin/env python
"""
eval_model.py  ──────────────────────────────────────────────────────────────
Evaluate a trained network and, for every mis‑classified example, generate
a Grad‑CAM heat‑map showing the spatial regions that most strongly pushed
the network toward its (incorrect) prediction.

Folder layout produced
──────────────────────
Reports/
└── Misclassified_CAMs/
    ├── img_00001_pred=dog_true=cat.png
    ├── img_00017_pred=truck_true=car.png
    └── …

Author : 2025‑07‑26
-----------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from util.device_detector import DeviceDetector


# ────────────────────────────────────────────────────────────────────────────
# 1.  Utilities
# ────────────────────────────────────────────────────────────────────────────


class GradCAM:
    """
    Minimal Grad‑CAM implementation for a *single* convolutional layer.

    Usage
    -----
    cam = GradCAM(model, target_layer)
    heatmap = cam(image_tensor, target_class)
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = self._get_last_conv_layer()

        self._features: List[torch.Tensor] = []
        self._gradients: List[torch.Tensor] = []

        # Register hooks
        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    # forward‑hook
    def _save_features(
            self,
            _: torch.nn.Module,
            __: Tuple[torch.Tensor],
            output: torch.Tensor,
    ) -> None:
        self._features.append(output.detach())

    # backward‑hook
    def _save_gradients(
            self,
            _: torch.nn.Module,
            grad_in: Tuple[torch.Tensor],
            grad_out: Tuple[torch.Tensor],
    ) -> None:
        self._gradients.append(grad_out[0].detach())

    def __call__(
            self, input: torch.Tensor, class_idx: int, normalize: bool = True
    ) -> Image.Image:
        """
        Parameters
        ----------
        input : (C,H,W) or (1,C,H,W) tensor
        class_idx : int
        normalize : bool
        """

        # 1. Handle Dimensions: Separate input for Model (4D) and Visualization (3D)
        if input.ndim == 4:
            # Input is (B, C, H, W) - use as is for model, squeeze for PIL
            input_model = input
            input_visual = input.squeeze(0)
        else:
            # Input is (C, H, W) - unsqueeze for model, use as is for PIL
            input_model = input.unsqueeze(0)
            input_visual = input

        # 2. Prepare Visualization Image (Must be 3D)
        img_tensor = self._denormalize(input_visual)  # (C,H,W)
        img_pil = to_pil_image(img_tensor.cpu())  # Ensure CPU for PIL compatibility

        # 3. Prepare Model Input (Must be 4D)
        input_tensor = DeviceDetector().to(input_model)

        self.model = DeviceDetector().to(self.model)
        self.model.eval()

        # Ensure gradients are tracked for the input to keep the graph alive
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(input_tensor)
            score = logits[:, class_idx].sum()

            self.model.zero_grad(set_to_none=True)
            score.backward()

        # Obtain data
        gradients = self._gradients.pop()  # (1,C,H',W')
        activations = self._features.pop()  # (1,C,H',W')

        # Global‑average‑pool the gradients -> weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1,1,H',W')
        cam = torch.relu(cam)  # discard negative contributions

        # Normalize to [0,1]
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        heat = cam.squeeze().cpu().numpy()

        if normalize:
            # Avoid division by zero
            heat -= heat.min()
            heat /= (heat.max() + 1e-8)

        return self._overlay_heatmap(img_pil, heat)

    def _get_last_conv_layer(self) -> torch.nn.Module:
        """
        Heuristically find *the last convolutional layer* in the model.
        Adjust if your architecture is unusual.
        """

        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module

        raise RuntimeError("No convolutional layer found for Grad‑CAM.")

    @staticmethod
    def _overlay_heatmap(
            img: Image.Image,
            heatmap: np.ndarray,
            alpha: float = 0.5,
            colormap: str = "jet",
    ) -> Image.Image:
        """
        Overlay a heat‑map (H,W) onto a PIL image. Returns a new PIL image (RGBA).
        """
        # colourise heatmap
        cmap = plt.get_cmap(colormap)
        coloured_hm = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H,W,3)
        hm_img = Image.fromarray(coloured_hm).resize(img.size, resample=Image.BICUBIC)

        # Convert both to RGBA
        return Image.blend(
            img.convert("RGBA"), hm_img.convert("RGBA"), alpha=alpha
        )

    @staticmethod
    def _denormalize(
            tensor: torch.Tensor,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> torch.Tensor:
        """
        Undo ImageNet normalization so colours look correct when converted to PIL.
        """
        mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
        std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
        return tensor * std_t + mean_t
