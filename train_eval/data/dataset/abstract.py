import logging
from abc import ABC, abstractmethod
from typing import Tuple, Sequence

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class AbstractDataset(Dataset, ABC):
    def __init__(self, data_augment: bool, num_classes: int) -> None:
        self._data_augment = data_augment
        self._num_classes = num_classes

    @property
    def data_augment(self) -> bool:
        return self._data_augment

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        pass

    @abstractmethod
    def get_labels(self) -> torch.Tensor:
        pass

    def get_class_weights(self, labels: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, Sequence[int]]:
        # Use CPU int64 for bincount; move back to original device at return
        labels_cpu = labels.to(dtype=torch.int64, device="cpu")

        # Validate labels are within 0..num_classes-1
        max_id = int(labels_cpu.max().item())
        min_id = int(labels_cpu.min().item())
        if min_id < 0 or max_id >= self._num_classes:
            raise ValueError(
                f"Label ids must be in [0, {self._num_classes - 1}] but got min={min_id}, max={max_id}. "
                "Either remap labels to contiguous ids or pass actual label ids in the constructor."
            )

        # Size everything by the constructor-declared class space
        num_slots = self._num_classes

        # True counts (no clamp); mask absent classes when computing weights
        counts: torch.Tensor = torch.bincount(labels_cpu, minlength=num_slots)

        # Inverse-frequency weights with epsilon; compute only over present classes
        present = counts > 0
        weights = torch.zeros(num_slots, dtype=torch.float32)
        weights[present] = 1.0 / (counts[present].to(dtype=torch.float32) + eps)

        # Normalize weights over present classes to keep scale stable
        if present.any():
            weights[present] = weights[present] / weights[present].mean()

        # Per-class sample counts (list[int]) sized by constructor classes
        samples_per_classes = [0] * num_slots
        for l in labels_cpu.tolist():
            samples_per_classes[l] += 1

        logging.info("Class counts : %s", counts.tolist())
        logging.info("Class weights: %s", weights.tolist())

        return weights.to(labels.device), samples_per_classes

    def get_sampler(self) -> Tuple[WeightedRandomSampler, torch.Tensor, Sequence[int]]:
        labels = self.get_labels()
        class_weights, samples_per_classes = self.get_class_weights(labels)
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(labels),
            replacement=True
        )

        return sampler, class_weights, samples_per_classes
