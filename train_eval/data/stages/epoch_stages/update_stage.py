import logging
from typing import Type, Dict

from data.loss.loss import Loss
from data.model_tag import ModelTag
from data.stages.epoch_stages.base_stage import EpochStageBase


class EpochUpdateStage(EpochStageBase):
    def __init__(self, manager: 'TrainManager'):
        self.epoch: int = manager.epoch
        self.model: 'Model' = manager.model
        self.data: 'Data' = manager.data
        self.stage_manager: 'StageManager' = manager.stage_manager
        self.stage: 'Stage' = manager.stage_manager.get_current_stage()
        self.stats: 'Stats' = manager.stats
        self.checkpoint_manager: 'CheckpointManager' = manager.checkpoint_manager

    def run(self) -> bool:
        if self.stage.stage.eval or not self.stage.stage.trained:
            return self.stage.stage.get_action_completed()

        self._save_best_checkpoint()

        losses: Dict[ModelTag, Loss] = {}
        f1s: Dict[ModelTag, Loss] = {}
        for item in self.data.get_evaluation_tags([self.stage.stage.domain]):
            model_tag = ModelTag(item, self.stage.stage.domain)
            result = self.stats.get_last_eval_stats(model_tag)
            if result is not None:
                losses[model_tag] = result.loss
                f1s[model_tag] = result.metrics.f1

        self.stage_manager.get_current_stage().stage.step_schedulers(self.model, self.stats)

        self.stage_manager.update_scheduler(self.epoch, losses, f1s)
        return self.stage.stage.get_action_completed()

    @property
    def next_substage(self) -> Type[EpochStageBase] | None:
        return None

    def _save_best_checkpoint(self):
        for model_tag in self.stats.best_f1.keys():
            best_f1 = self.stats.best_f1[model_tag]
            current_f1 = self.stats.get_last_eval_stats(model_tag).metrics.f1

            if best_f1 < current_f1:
                logging.info(
                    f'Epoch: {self.epoch} - {model_tag.tag} - New best F1: {(current_f1 * 100):.2f}% (previous: {(best_f1 * 100):.2f}%)')

                self.stats.best_f1[model_tag] = current_f1
                self.stats.best_f1_epoch[model_tag] = self.epoch
                self.checkpoint_manager.save(self.epoch, sub_folder=model_tag.best_model_tag)
            elif best_f1 == current_f1:
                self.checkpoint_manager.save(self.epoch,
                                             sub_folder=model_tag.best_model_tag)
