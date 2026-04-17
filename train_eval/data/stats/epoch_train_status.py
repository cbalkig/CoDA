from typing import Dict, Optional

from data.lambdas import Lambdas


class EpochTrainStatus:
    def __init__(
            self,
            lambdas: Lambdas,
            mixup_prob: float,
            dropouts: Dict[str, float],
            num_layers: Dict[str, int],
            pseudo_label_counts: Optional[Dict[str, int]] = None,
            correct_pseudo_label_counts: Optional[Dict[str, int]] = None,
            incorrect_pseudo_label_counts: Optional[Dict[str, int]] = None,
            correct_pseudo_labels: int = 0,
            incorrect_pseudo_labels: int = 0,
            pseudo_label_thresholds: Optional[Dict[str, float]] = None,
            thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lambdas = lambdas

        self.mixup_prob = mixup_prob
        self.dropouts = dropouts

        if pseudo_label_counts is None:
            pseudo_label_counts = {}

        if incorrect_pseudo_label_counts is None:
            incorrect_pseudo_label_counts = {}

        self.pseudo_label_counts = pseudo_label_counts
        self.correct_pseudo_label_counts = correct_pseudo_label_counts
        self.incorrect_pseudo_label_counts = incorrect_pseudo_label_counts

        self.correct_pseudo_labels = correct_pseudo_labels
        self.incorrect_pseudo_labels = incorrect_pseudo_labels
        self.pseudo_label_thresholds = pseudo_label_thresholds

        self.num_layers = num_layers
        self.thresholds = thresholds or {}

    def __str__(self) -> str:
        text = "Train Epoch Status:\n - Lambdas:\n\t{self._format_nested(self.lambdas)}\n"

        for attr, value in vars(self).items():
            if isinstance(value, float):
                text += f" - {attr.replace('_', ' ').title()}: {value:.10f}\n"
            elif isinstance(value, dict):
                text += f" - {attr.replace('_', ' ').title()}: {value}\n"
            else:
                text += f" - {attr.replace('_', ' ').title()}: {value}\n"
        return text
