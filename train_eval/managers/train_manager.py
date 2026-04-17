import logging
from collections import OrderedDict

from tqdm import tqdm

from data.configuration.configuration import Configuration
from data.dataset.data import Data
from data.stages.stages.source.train_source import TrainSourceStage
from data.types.domain_type import DomainType
from managers.operation_manager import OperationManager
from model.model import Model
from util.device_detector import DeviceDetector


class TrainManager(OperationManager):
    def __init__(self, model: Model, data: Data, config: Configuration):
        super().__init__(model, data, config, TrainSourceStage())

    def _log_progress_bar(self, pbar: tqdm):
        stats = OrderedDict()

        if self.stats.get_last_train_stats() is not None:
            train_stats = self.stats.get_last_train_stats()
            for domain in DomainType:
                try:
                    stats[f'{domain}_num_unfrozen_layers'] = (
                        f"{train_stats.status.num_layers[f'{domain}_feature_extractor_unfrozen']} / "
                        f"{train_stats.status.num_layers[f'{domain}_feature_extractor']}"
                    )
                    stats[f'{domain}_learning_rates'] = (
                        f"[FE]: {train_stats.learning_rates.get(f'{domain}_feature_extractor'):.6f} | "
                        f"[MLP]: {train_stats.learning_rates.get(f'{domain}_classifier'):.6f}"
                    )
                except KeyError:
                    pass

            pbar.set_postfix(stats, refresh=False)

        if len(self.stats.best_f1.keys()) > 0:
            text = " | ".join(
                f"[{model_tag.tag}]: {score * 100:.2f}% (Epoch {self.stats.best_f1_epoch[model_tag]})"
                for model_tag, score in self.stats.best_f1.items()
            )
            stats['best_f1'] = text
            pbar.set_postfix(stats, refresh=False)

        pbar.refresh()

        log_lines = ["Epoch stats:"]
        for key, value in stats.items():
            log_lines.append(f"  - {key}: {value}")
        logging.info("\n".join(log_lines))

    def _check_stage_finished(self, stage_changed: bool) -> bool:
        if ((stage_changed and self.stage_manager.stage_scheduler.check_termination(self.epoch)) or
                not self.stage_manager.get_current_stage().stage.continuous):
            return True

        return False

    def run(self):
        logging.warning(
            f'Will start training with epoch {self.epoch + 1} - {self.stage_manager.get_current_stage().stage.stage.name}')

        pbar = tqdm(
            total=None,
            position=0,
            leave=True,
            dynamic_ncols=True,
            desc=f"🚂 Train Epoch: {self.epoch}",
        )

        stage_changed = True
        at_least_one_epoch = False
        while True:
            self._log_progress_bar(pbar)

            if at_least_one_epoch:
                self.checkpoint_manager.save(self.epoch, sub_folder="checkpoint")

            stage_finished = self._check_stage_finished(stage_changed)
            if stage_finished:
                logging.warning(f'Epoch: {self.epoch} - No more improvement...')
                next_stage = self.stage_manager.move_to_next()
                if next_stage is not None:
                    self.switch_stage(next_stage, backup=at_least_one_epoch)
                else:
                    logging.warning(f'Epoch: {self.epoch} - Exit from execution.')
                    self.finish()
                    break
            else:
                if at_least_one_epoch:
                    logging.debug(
                        f'Epoch: {self.epoch} - Epochs since last state change: '
                        f'{self.epoch - self.stage_manager.get_last_state_change_epoch()}')

                self.epoch += 1
                pbar.update(1)
                pbar.set_description(f"🚂 Train Epoch: {self.epoch}")

                stage_changed = self._execute_substages()
                self.tensorboard.log_epoch_data(self.epoch, self.stats, self.data)

                at_least_one_epoch = True

        DeviceDetector().empty_cache()
