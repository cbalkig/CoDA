from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from data.variations.data_augmentation.abstract_transforms import AbstractTransform, AddGaussianNoise


class DataTransforms(AbstractTransform):
    """
    Baseline 3: 2D Random Patch Removal (Cutout / Random Erasing) — v7_cutout.

    Heavy Cutout augmentation is applied by stacking multiple large
    RandomErasing passes in normalised tensor space.  Each pass independently
    decides whether to fire and where to place the patch, so the model sees
    between 0 and `num_cutout_passes` blacked-out regions per image.

    Optionally a static-noise fill variant is enabled via `noise_fill=True`,
    which replaces the black value with per-patch Gaussian noise, simulating
    2D interior-mask noise over the object.

    Geometry is kept minimal (flip + light affine) so the ablation isolates
    the effect of patch removal rather than confounding it with heavy spatial
    distortions.
    """

    def __init__(
            self,
            image_size: int,
            aug_prob: float = 0.8,
            noise_p: float = 0.10,
            # Cutout / erasing knobs
            num_cutout_passes: int = 3,
            cutout_p: float = 0.70,
            cutout_scale_min: float = 0.10,
            cutout_scale_max: float = 0.40,
            noise_fill: bool = False,
    ) -> None:
        super().__init__(image_size, aug_prob)
        self.noise_p = float(noise_p)
        self.num_cutout_passes = num_cutout_passes
        self.cutout_p = cutout_p
        self.cutout_scale_min = cutout_scale_min
        self.cutout_scale_max = cutout_scale_max
        self.noise_fill = noise_fill

    def _aug_block(self) -> T.Compose:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),

            # Light affine only — keep spatial distortion minimal for this baseline
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

            # Brightness / contrast — no hue to keep colour neutral
            T.RandomApply([
                T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.0, hue=0.0)
            ], p=0.50),

            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))], p=0.05),
        ])

    def _cutout_passes(self) -> list[T.RandomErasing]:
        """Build `num_cutout_passes` independent heavy-erasing transforms."""
        fill_value = "random" if self.noise_fill else 0.0
        return [
            T.RandomErasing(
                p=self.cutout_p,
                scale=(self.cutout_scale_min, self.cutout_scale_max),
                ratio=(0.5, 2.0),   # near-square patches
                value=fill_value,
                inplace=False,
            )
            for _ in range(self.num_cutout_passes)
        ]

    def train_transforms(self) -> T.Compose:
        return T.Compose([
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.60, 1.0),
                ratio=(0.80, 1.25),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),

            T.RandomApply([self._aug_block()], p=self.aug_prob),

            T.ToTensor(),
            AddGaussianNoise(sigma_min=0.0, sigma_max=0.02, p=self.noise_p),

            T.Normalize(self.mean, self.std),

            # ── Heavy Cutout ─────────────────────────────────────────────────
            # Multiple independent passes; each fires with probability
            # `cutout_p` and drops a large square region (10–40 % of area).
            # Set noise_fill=True to replace black with interior static noise.
            *self._cutout_passes(),
            # ────────────────────────────────────────────────────────────────
        ])
