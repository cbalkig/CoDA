from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from data.variations.data_augmentation.abstract_transforms import AbstractTransform, AddGaussianNoise


class DataTransforms(AbstractTransform):
    """
    Sculpture-aware transform set (v5).

    Differences vs. aggressive HDR pipelines (v2):
    - Tighter crop range (scale=(0.60, 1.0), ratio=(0.80, 1.25)) to respect silhouette.
    - Reduced geometry jitter; explicit small rotation/affine/perspective with lower p.
    - Photometric ops avoid hue/saturation; use only brightness/contrast (+ light autocontrast).
    - Very light blur/noise; no posterize/solarize; no border erase.
    - Small random erasing patches with black value to blend with dark backgrounds.
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

            # Gentle geometry
            T.RandomApply([
                T.RandomRotation(
                    degrees=15,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb
                )
            ], p=0.25),

            T.RandomApply([
                T.RandomPerspective(
                    distortion_scale=0.15, p=1.0,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb
                )
            ], p=0.20),

            T.RandomApply([
                T.RandomAffine(
                    degrees=8,
                    translate=(0.03, 0.03),
                    scale=(0.92, 1.08),
                    shear=(-3, 3, -3, 3),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb,
                )
            ], p=0.40),

            # Monochrome-safe photometrics
            T.RandomApply([
                T.ColorJitter(brightness=0.20, contrast=0.25, saturation=0.0, hue=0.0)
            ], p=0.60),
            T.RandomAutocontrast(p=0.10),
            T.RandomAdjustSharpness(sharpness_factor=1.25, p=0.05),

            # Very light blur
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))], p=0.05),
        ])

    def train_transforms(self) -> T.Compose:
        return T.Compose([
            # Tighter crop/ratio to preserve silhouette integrity
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.60, 1.0),
                ratio=(0.80, 1.25),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),

            # Pack the core block under a single gate (keeps parity with v2/v3 style)
            T.RandomApply([self._aug_block()], p=self.aug_prob),

            # Tensor & low-level noise before normalization
            T.ToTensor(),
            AddGaussianNoise(sigma_min=0.0, sigma_max=0.02, p=self.noise_p),

            # Normalize as in other variants
            T.Normalize(self.mean, self.std),

            # Small black cutouts; avoids colored blotches on dark backgrounds
            T.RandomErasing(
                p=0.30,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            ),
        ])
