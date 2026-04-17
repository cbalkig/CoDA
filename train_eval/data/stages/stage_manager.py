import logging
from typing import Dict

from data.loss.loss import Loss
from data.model_tag import ModelTag
from data.stages.stage import Stage
from data.stages.stage_base import StageBase
from data.stages.stage_history import StageHistory
from data.stages.stage_types import Stages
from data.stats.epoch_stage_status import EpochStageStatus
from managers.nns import NetworkStabilityScheduler


class StageManager:
    stage_scheduler: NetworkStabilityScheduler
    history: StageHistory

    def __init__(self) -> None:
        self.history = StageHistory()
        self.stage_scheduler = NetworkStabilityScheduler()

    def update_scheduler(self, epoch: int, losses: Dict[ModelTag, Loss], f1s: Dict[ModelTag, float]) -> None:
        current_stage = self.history.get_current_stage().stage
        loss, f1 = current_stage.get_improvement(losses, f1s)
        self.stage_scheduler.step(epoch, loss, f1)

    def set_stage(self, epoch: int, stage: StageBase) -> Stage:
        previous_stage_name = self.history.get_current_stage().stage.stage.name if self.history.history else None

        if previous_stage_name is None:
            logging.warning(f'Epoch: {epoch} - Stage set to {stage.stage.name} - Switching...')
            self.history.add(stage, epoch)
        elif previous_stage_name != stage.stage.name:
            logging.warning(
                f'Epoch: {epoch} - Stage set from {previous_stage_name} to {stage.stage.name} - Switching...')
            self.history.add(stage, epoch)

        self.stage_scheduler.change_stage(epoch)
        return self.history.get_current_stage()

    def move_to_next(self) -> None | StageBase:
        current_stage = self.history.get_current_stage().stage
        next_stage_type = current_stage.next_stage

        if next_stage_type:
            return next_stage_type()
        else:
            logging.warning("No next stage available for the current stage.")

        return None

    def get_stage_history(self) -> StageHistory:
        return self.history

    def get_current_stage(self) -> None | Stage:
        return self.get_stage_history().get_current_stage()

    def get_current_stage_type(self) -> Stages:
        return self.get_stage_history().get_current_stage_type()

    def get_stage_status(self) -> EpochStageStatus:
        return self.stage_scheduler.get_stage_status()

    def get_last_state_change_epoch(self) -> int:
        return self.get_current_stage().epoch
