from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from data.variations.data_augmentation.abstract_transforms import AbstractTransform, AddGaussianNoise


class DataTransforms(AbstractTransform):
    """
    Baseline 2: 2D Grayscale / Desaturation (Post-hoc Color Removal) — v6_gray.

    Fully-textured images are converted to grayscale using a standard 2D image
    processing operation (T.Grayscale), simulating colour-free input regardless
    of the original rendering mode (Studio, Solid Colour, etc.).

    The grayscale conversion is applied deterministically *before* tensor
    conversion so that colour information is removed at the PIL stage.
    Standard geometric augmentations are kept to maintain training robustness.
    No hue / saturation jitter is applied because the image is already
    desaturated.
    """

    def __init__(
            self,
            image_size: int,
            aug_prob: float = 0.8,
            noise_p: float = 0.10,
    ) -> None:
        super().__init__(image_size, aug_prob)
        self.noise_p = float(noise_p)

    def _aug_block(self) -> T.Compose:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),

            # Gentle geometry — same as v5
            T.RandomApply([
                T.RandomRotation(
                    degrees=15,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=0,  # fill with black for grayscale
                )
            ], p=0.25),

            T.RandomApply([
                T.RandomPerspective(
                    distortion_scale=0.15, p=1.0,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=0,
                )
            ], p=0.20),

            T.RandomApply([
                T.RandomAffine(
                    degrees=8,
                    translate=(0.03, 0.03),
                    scale=(0.92, 1.08),
                    shear=(-3, 3, -3, 3),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=0,
                )
            ], p=0.40),

            # Brightness / contrast only — no hue or saturation on a gray image
            T.RandomApply([
                T.ColorJitter(brightness=0.20, contrast=0.25, saturation=0.0, hue=0.0)
            ], p=0.60),
            T.RandomAutocontrast(p=0.10),
            T.RandomAdjustSharpness(sharpness_factor=1.25, p=0.05),

            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))], p=0.05),
        ])

    def train_transforms(self) -> T.Compose:
        return T.Compose([
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.60, 1.0),
                ratio=(0.80, 1.25),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),

            # ── Post-hoc colour removal ──────────────────────────────────────
            # Convert to grayscale and replicate across 3 channels so that
            # downstream layers expecting RGB input continue to work unchanged.
            T.Grayscale(num_output_channels=3),
            # ────────────────────────────────────────────────────────────────

            T.RandomApply([self._aug_block()], p=self.aug_prob),

            T.ToTensor(),
            AddGaussianNoise(sigma_min=0.0, sigma_max=0.02, p=self.noise_p),

            T.Normalize(self.mean, self.std),

            T.RandomErasing(
                p=0.30,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            ),
        ])
