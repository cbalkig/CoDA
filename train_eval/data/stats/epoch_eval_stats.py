from typing import List

from data.loss.loss import Loss
from data.metrics import Metrics
from data.stats.epoch_stats import EpochStats


class EpochEvalStats(EpochStats):
    def __init__(self, epoch: int, loss: Loss, labels: List[int], predictions: List[int], metrics: Metrics) -> None:
        self.epoch = epoch
        self.loss = loss
        self.labels = labels
        self.predictions = predictions
        self.metrics = metrics

    def __str__(self) -> str:
        return (
            f"Epoch: {self.epoch}\n"
            f" - Losses:\n\t{self._format_nested(self.loss)}\n"
            f" - Metrics:\n\t{self._format_nested(self.metrics)}\n"
            f" - Labels: {self.labels[:10]}{'...' if len(self.labels) > 10 else ''}\n"
            f" - Predictions: {self.predictions[:10]}{'...' if len(self.predictions) > 10 else ''}\n"
        )
