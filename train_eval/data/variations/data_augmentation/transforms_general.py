from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from data.variations.data_augmentation.abstract_transforms import AbstractTransform


class DataTransforms(AbstractTransform):
    """
    A lighter, efficient, and general data transformation pipeline using RandAugment.
    """

    def __init__(
            self,
            image_size: int,
            # RandAugment parameters (standard defaults)
            rand_aug_num_ops: int = 2,  # Number of operations applied sequentially
            rand_aug_magnitude: int = 9,  # Magnitude of operations (scale 0-30)
            # Regularization
            erasing_p: float = 0.25,
    ) -> None:
        # Assuming the base class initializes self.image_size, self.mean, self.std.
        # We set aug_prob to 0.0 as we are not using the original probabilistic block structure.
        super().__init__(image_size, aug_prob=0.0)

        self.rand_aug_num_ops = rand_aug_num_ops
        self.rand_aug_magnitude = rand_aug_magnitude
        self.erasing_p = erasing_p

    def _aug_block(self) -> T.Compose:
        pass

    def train_transforms(self) -> T.Compose:
        return T.Compose([
            # 1. Initial Crop and Resize
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),  # Standard ratio
                # Using BILINEAR is significantly faster than BICUBIC
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),

            T.RandomHorizontalFlip(p=0.5),

            # 2. RandAugment (Replaces the entire original _aug_block)
            T.RandAugment(
                num_ops=self.rand_aug_num_ops,
                magnitude=self.rand_aug_magnitude,
                interpolation=InterpolationMode.BILINEAR
            ),

            # 3. Finalization and Regularization
            T.ToTensor(),
            T.Normalize(self.mean, self.std),

            # 4. Random Erasing (Cutout)
            T.RandomErasing(
                p=self.erasing_p,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            ),
        ])
