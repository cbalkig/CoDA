from typing import Dict

from data.loss.losses import Losses
from data.stats.epoch_stage_status import EpochStageStatus
from data.stats.epoch_stats import EpochStats
from data.stats.epoch_train_status import EpochTrainStatus


class EpochTrainStats(EpochStats):
    def __init__(self, epoch: int, learning_rates: Dict[str, float], losses: Losses,
                 status: EpochTrainStatus, stage_status: EpochStageStatus) -> None:
        self.epoch = epoch
        self.learning_rates = learning_rates
        self.losses = losses
        self.status = status
        self.stage_status = stage_status

    def __str__(self) -> str:
        return (
            f"Epoch: {self.epoch}\n"
            f" - Learning Rates:\n\t{self._format_nested(self.learning_rates)}"
            f" - Losses:\n\t{self._format_nested(self.losses)}"
            f" - Status:\n\t{self._format_nested(self.status)}"
            f" - Stage Status:\n\t{self._format_nested(self.stage_status)}"
        )
