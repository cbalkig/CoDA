# transforms.py
from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from data.variations.data_augmentation.abstract_transforms import AbstractTransform


class DataTransforms(AbstractTransform):
    def __init__(self, image_size: int, aug_prob: float = 0.8) -> None:
        super().__init__(image_size, aug_prob)

    def _aug_block(self) -> T.Compose:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.RandomPerspective(distortion_scale=0.2, p=1.0, interpolation=InterpolationMode.BICUBIC,
                                    fill=self.fill_rgb),
            ], p=0.4),
            T.RandomApply([
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.90, 1.10),
                    shear=(-5, 5, -5, 5),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill_rgb,
                )
            ], p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ], p=0.8),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))], p=0.15),
        ])

    def train_transforms(self) -> T.Compose:
        return T.Compose([
            # Always crop once to avoid extra resizes
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.7, 1.0),  # a bit wider for robustness
                ratio=(0.75, 1.33),  # allow aspect variation
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomApply([self._aug_block()], p=self.aug_prob),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
            # Erase in normalized space so value=0 really means "neutral"
            T.RandomErasing(
                p=0.3,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            ),
        ])
