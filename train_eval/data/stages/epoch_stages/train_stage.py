import logging
import time
import traceback
from typing import Type, Optional, Any

from torch import OutOfMemoryError

from configs.base.configs import Configs
from data.indomain_lambdas import InDomainLambdas
from data.lambdas import Lambdas
from data.loss.losses import Losses
from data.stages.epoch_stages.base_stage import EpochStageBase
from data.stages.epoch_stages.eval_stage import EpochEvalStage
from data.stats.epoch_train_stats import EpochTrainStats
from data.stats.epoch_train_status import EpochTrainStatus


class EpochTrainStage(EpochStageBase):
    def __init__(self, manager: 'TrainManager'):
        self.model: 'Model' = manager.model
        self.epoch: int = manager.epoch
        self.data: 'Data' = manager.data
        self.stage: 'Stage' = manager.stage_manager.get_current_stage()
        self.stage_type: 'Stages' = manager.stage_manager.get_current_stage_type()
        self.stage_manager: 'StageManager' = manager.stage_manager
        self.dropout_manager: 'DropoutManager' = manager.dropout_manager
        self.mixup_manager: 'MixupManager' = manager.mixup_manager
        self.stats: 'Stats' = manager.stats
        self.total_loss: Optional[Losses] = None
        self.train_epoch_status: Optional[EpochTrainStatus] = None
        self.lambdas: Lambdas = None

    def run(self) -> None:
        self.lambdas = Lambdas(InDomainLambdas(Configs().training.lambdas['in_domain']['classification']),
                               Configs().training.lambdas['domains']['source'],
                               Configs().training.lambdas['domains']['target'])
        self.total_loss = self.stage.stage.get_total_loss(self)

        params: dict[str, Any] = self.stage.stage.get_train_status()

        self.train_epoch_status = EpochTrainStatus(self.lambdas, self.mixup_manager.mixup_prob,
                                                   dropouts=self.model.get_dropouts(),
                                                   num_layers=self.model.get_num_layers(), **params)

        while True:
            try:
                self.stage.stage.run(self)
                break
            except OutOfMemoryError as e:
                current_stack = "".join(traceback.format_stack())

                logging.warning(
                    f'Out of memory error: {e}\n'
                    f'Caller Stack Trace:\n{current_stack}'
                )

                time.sleep(60)

        learning_rates = self.model.get_learning_rates()
        stage_epoch_status = self.stage_manager.get_stage_status()
        train_stats = EpochTrainStats(self.epoch, learning_rates, self.total_loss,
                                      self.train_epoch_status,
                                      stage_epoch_status)
        self.stats.add_training_stats(train_stats)

    @property
    def next_substage(self) -> Type[EpochStageBase] | None:
        return EpochEvalStage
