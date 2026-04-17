import logging
from typing import Dict, List

from data.file.path import StoragePath
from data.model_tag import ModelTag
from data.stats.epoch_eval_stats import EpochEvalStats
from data.stats.epoch_train_stats import EpochTrainStats
from util.file_util import FileUtil


class Stats:
    def __init__(self) -> None:
        self.train_stats: List[EpochTrainStats] = []
        self.eval_stats: Dict[ModelTag, EpochEvalStats] = {}
        self.best_f1: Dict[ModelTag, float] = {}
        self.best_f1_epoch: Dict[ModelTag, int] = {}

    def add_training_stats(self, stats: EpochTrainStats) -> None:
        if stats is None or stats.epoch is None:
            return

        previous_stats = None

        if stats.epoch > 1:
            previous_stats = self.get_last_train_stats()

        self.train_stats.append(stats)

        if previous_stats is not None:
            for key in stats.learning_rates:
                if key in previous_stats.learning_rates:
                    if stats.learning_rates[key] != previous_stats.learning_rates[key]:
                        logging.debug(
                            f'Epoch: {stats.epoch} - {key} - Learning rate {previous_stats.learning_rates[key]:.6f} changed to {stats.learning_rates[key]:.6f}')

    def add_evaluation_stats(self, model_tag: ModelTag, stats: EpochEvalStats) -> None:
        self.eval_stats[model_tag] = stats

        if model_tag not in self.best_f1:
            self.best_f1[model_tag] = -1

        if model_tag not in self.best_f1_epoch:
            self.best_f1_epoch[model_tag] = -1

    def get_last_train_stats(self) -> EpochTrainStats | None:
        if len(self.train_stats) == 0:
            return None

        return self.train_stats[-1]

    def get_last_eval_stats(self, model_tag: ModelTag) -> EpochEvalStats | None:
        return self.eval_stats[model_tag] if model_tag in self.eval_stats else None

    def get_best_f1(self, model_tag: ModelTag) -> float:
        return self.best_f1[model_tag]

    def save(self, file_path: StoragePath):
        FileUtil().dump(self.__dict__, file_path)

    @classmethod
    def load(cls, file_path: StoragePath) -> "Stats":
        data = FileUtil().load(file_path, weights_only=False)
        obj = cls()
        obj.__dict__.update(data)
        return obj
