from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from data.variations.data_augmentation.abstract_transforms import AbstractTransform, AddGaussianNoise, RandomBorderErase

try:
    from torchvision.transforms import ElasticTransform as TVElasticTransform  # type: ignore
except Exception:
    TVElasticTransform = None


class DataTransforms(AbstractTransform):
    def __init__(
            self,
            image_size: int,
            aug_prob: float = 0.8,
            rotation_deg: int = 20,
            use_elastic_if_available: bool = True,
            noise_p: float = 0.25,
            border_erase_p: float = 0,
    ) -> None:
        super().__init__(image_size, aug_prob)
        self.rotation_deg = rotation_deg
        self.use_elastic_if_available = use_elastic_if_available
        self.noise_p = noise_p
        self.border_erase_p = border_erase_p

    def _aug_block(self) -> T.Compose:
        ops = [
            T.RandomHorizontalFlip(p=0.5),

            T.RandomApply([
                T.RandomRotation(
                    degrees=self.rotation_deg,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb,
                )
            ], p=0.30),

            T.RandomApply([
                T.RandomPerspective(
                    distortion_scale=0.2, p=1.0,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb
                )
            ], p=0.30),  # ↓ from 0.40

            T.RandomApply([
                T.RandomAffine(
                    degrees=10, translate=(0.05, 0.05),
                    scale=(0.90, 1.10), shear=(-5, 5, -5, 5),
                    interpolation=InterpolationMode.BICUBIC, fill=self.fill_rgb,
                )
            ], p=0.50),

            T.RandomApply([
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06)
            ], p=0.80),

            T.RandomGrayscale(p=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))], p=0.05),  # ↓ from 0.15

            T.RandomApply([T.RandomPosterize(bits=3)], p=0.02),  # ↓ from 0.10
            T.RandomApply([T.RandomSolarize(threshold=128)], p=0.02),  # ↓ from 0.10
            T.RandomAutocontrast(p=0.10),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.10),
        ]

        if self.use_elastic_if_available and TVElasticTransform is not None:
            ops.insert(3, T.RandomApply([
                TVElasticTransform(
                    alpha=50.0, sigma=6.0,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb,
                )
            ], p=0.20))

        return T.Compose(ops)

    def train_transforms(self) -> T.Compose:
        return T.Compose([
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.55, 1.0),  # ↑ was (0.50, 1.0)
                ratio=(0.70, 1.40),
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            ),
            T.RandomApply([self._aug_block()], p=self.aug_prob),

            T.ToTensor(),
            AddGaussianNoise(sigma_min=0.0, sigma_max=0.03, p=self.noise_p),

            T.Normalize(self.mean, self.std),

            T.RandomErasing(  # keep p & scale but use random-colored erasing
                p=0.50, scale=(0.02, 0.20), ratio=(0.3, 3.3),
                value="random", inplace=False
            ),
            RandomBorderErase(p=self.border_erase_p, max_frac=0.12, value=0.0),
        ])
