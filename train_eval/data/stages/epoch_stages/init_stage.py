from typing import Type

from data.stages.epoch_stages.base_stage import EpochStageBase
from data.stages.epoch_stages.train_stage import EpochTrainStage


class EpochInitStage(EpochStageBase):
    def __init__(self, manager: "TrainManager") -> None:
        super().__init__()

        self.model: 'Model' = manager.model
        self.epoch: int = manager.epoch
        self.stage: 'Stage' = manager.stage_manager.get_current_stage()
        self.stage_scheduler: 'StageScheduler' = manager.stage_manager.stage_scheduler
        self.mixup_manager: 'MixupManager' = manager.mixup_manager
        self.dropout_manager: 'DropoutManager' = manager.dropout_manager
        self.pseudo_label_manager: 'PseudoLabelManager' = manager.pseudo_label_manager
        self.data: 'Data' = manager.data

    def run(self) -> None:
        self.stage.stage.init_epoch(self)

    @property
    def next_substage(self) -> Type[EpochStageBase] | None:
        return EpochTrainStage
