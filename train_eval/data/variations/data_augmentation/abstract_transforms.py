import random
from abc import abstractmethod, ABC

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class AbstractTransform(ABC):
    IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __init__(self, image_size: int, aug_prob: float = 0.8) -> None:
        self.image_size = image_size
        self.mean, self.std = self.IMAGENET_STATS
        self.aug_prob = aug_prob
        # PIL fill expects 0–255 ints (for 3 channels)
        self.fill_rgb = tuple(int(m * 255) for m in self.mean)

    @abstractmethod
    def _aug_block(self) -> T.Compose:
        pass

    @abstractmethod
    def train_transforms(self) -> T.Compose:
        pass

    def test_transforms(self) -> T.Compose:
        resize_short = int(round(self.image_size / 0.875))
        return T.Compose([
            T.Resize(resize_short, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma_min: float, sigma_max: float, p: float) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p > 0 and random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            noise = torch.randn_like(x) * sigma
            x = (x + noise).clamp(0.0, 1.0)
        return x


class RandomBorderErase(torch.nn.Module):
    def __init__(self, p: float, max_frac: float, value: float) -> None:
        super().__init__()
        self.p = p
        self.max_frac = max_frac
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() >= self.p:
            return x
        c, h, w = x.shape
        borders = ["top", "bottom", "left", "right"]
        random.shuffle(borders)
        k = random.choice([1, 2])
        for b in borders[:k]:
            frac = random.uniform(0.02, self.max_frac)
            if b == "top":
                hh = int(h * frac)
                x[:, :hh, :] = self.value
            elif b == "bottom":
                hh = int(h * frac)
                x[:, h - hh:, :] = self.value
            elif b == "left":
                ww = int(w * frac)
                x[:, :, :ww] = self.value
            elif b == "right":
                ww = int(w * frac)
                x[:, :, w - ww:] = self.value
        return x
