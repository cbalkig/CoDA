import logging
from abc import ABC, abstractmethod
from typing import Optional

from configs.base.configs import Configs
from data.configuration.configuration import Configuration
from data.dataset.data import Data
from data.file.path import StoragePath
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.stage_base import StageBase
from data.stages.stage_manager import StageManager
from data.stages.stages.eval.eval_model import EvaluateModelTypeStage
from data.stats.stats import Stats
from data.types.operation_type import OperationType
from managers.checkpoint_manager import CheckpointManager
from managers.tensorboard.report import export
from managers.tensorboard_logger import TensorboardLogger
from model.managers.dropout_manager import DropoutManager
from model.managers.mixup_manager import MixupManager
from model.managers.pseudo_label_manager import PseudoLabelManager
from model.model import Model
from util.device_detector import DeviceDetector
from util.file_util import FileUtil
from util.helper_funcs import HelperFuncs
from util.implementations import find_implementations_in_package


class OperationManager(ABC):
    def __init__(self, model: Model, data: Data, config: Configuration, initial_stage: StageBase,
                 initial_epoch: Optional[int] = 0, ):
        self._raise_fd_limit()

        self.model = model
        self.model_folder_path = config.model.path
        self.tensorboard_folder_path = config.tensorboard_folder_path
        self.data = data
        self.source_cross_val_k = config.source_cross_val_k
        self.target_cross_val_k = config.target_cross_val_k

        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.model_id = config.model.model_id
        self.operation = config.operation
        self.resume = False

        self.stats = Stats()

        self.base_model_path = config.base_model.path if config.base_model is not None else None
        self.stage_manager = StageManager()
        self.checkpoint_manager = CheckpointManager(config.log_file_path, config.tensorboard_folder_path, self, data)
        self.dropout_manager = DropoutManager()

        self.thresholds = {
            'hard_positive_source': 0,
            'hard_positive_target': 0
        }

        self.mixup_manager = MixupManager()
        self.pseudo_label_manager = PseudoLabelManager()

        if self.base_model_path is not None:
            logging.warning(f'Will load base model: {config.base_model.model_id}')
            logging.info(f"Loading base model from: {self.base_model_path}")
            base_model_path = config.model.path.join('base_model')
            FileUtil().unzip_file(StoragePath(f'{self.base_model_path}.zip'), base_model_path)
            logging.info(f'Copied base model from {self.base_model_path} to {base_model_path}')

            if Configs().general.operation == OperationType.EVALUATION or Configs().pretrained.weights_only:
                self.checkpoint_manager.load(sub_folder=Configs().pretrained.model, folder=base_model_path)
                logging.warning(f'Reloading pretrained weights from {base_model_path}')
            else:
                self.checkpoint_manager.load(folder=base_model_path)
                logging.warning(f'Reloading epoch of {self.initial_epoch} model from {base_model_path}')
                self.epoch = self.initial_epoch
                self.resume = True

            FileUtil().delete_directory(base_model_path)

        new_stage: Optional[StageBase] = None
        if Configs().general.stage is not None:
            stage = Configs().general.stage

            import data.stages.stages as stages
            classes = find_implementations_in_package(StageBase, stages)

            for cls in classes:
                impl = cls()
                if impl.stage == stage:
                    new_stage = impl
                    break

            if new_stage is None:
                raise Exception(f'No stage found for {stage}')

        if self.resume:
            if new_stage is not None:
                self.switch_stage(new_stage, preprocess=(isinstance(new_stage, EvaluateModelTypeStage)), backup=False)
            else:
                self.switch_stage(initial_stage, preprocess=(isinstance(initial_stage, EvaluateModelTypeStage)),
                                  backup=False)
        else:
            if new_stage is not None:
                self.switch_stage(new_stage)
            else:
                self.switch_stage(initial_stage)

        self.tensorboard = TensorboardLogger(self.tensorboard_folder_path)
        self._finished = False

    @staticmethod
    def _raise_fd_limit(target_soft: int = 4096):
        try:
            import resource  # POSIX only
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            new_soft = min(target_soft, hard if hard > 0 else target_soft)
            if soft < new_soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                logging.debug(f'Raised RLIMIT_NOFILE soft limit: {soft} -> {new_soft}')
        except Exception as e:
            logging.warning(f'Could not raise RLIMIT_NOFILE: {e}')

    def switch_stage(self, new_stage: StageBase, preprocess: bool = True, backup: bool = True) -> None:
        current_stage = self.stage_manager.get_current_stage()
        if current_stage is not None and new_stage.stage != current_stage.stage.stage:
            logging.info(f'Saving old stage to: {self.stage_manager.get_current_stage()}')
            if backup:
                self._backup()

        self.stage_manager.set_stage(self.epoch, new_stage)

        if preprocess:
            logging.warning(f'Stage set to {self.stage_manager.get_current_stage()} - Preprocessing...')
            self.stage_manager.get_current_stage().stage.preprocess(self.epoch, self.model, self.data,
                                                                    self.dropout_manager, self.mixup_manager,
                                                                    self.pseudo_label_manager)

    def _backup(self) -> None:
        if self.operation != OperationType.EVALUATION:
            export(self.tensorboard_folder_path.path)

            if self.source_cross_val_k is not None and self.source_cross_val_k > 0:
                if self.target_cross_val_k is not None and self.target_cross_val_k > 0:
                    backup_path = Configs().storage.backup_folder.join(
                        f'{Configs().general.tag}_{self.source_cross_val_k}_{self.target_cross_val_k}_{self.stage_manager.get_current_stage().stage.stage.value}.zip')
                else:
                    backup_path = Configs().storage.backup_folder.join(
                        f'{Configs().general.tag}_{self.source_cross_val_k}_{self.stage_manager.get_current_stage().stage.stage.value}.zip')
            else:
                backup_path = Configs().storage.backup_folder.join(
                    f'{Configs().general.tag}_{self.stage_manager.get_current_stage().stage.stage.value}.zip')

            logging.warning(f'Backing up {self.model_folder_path} to {backup_path}')

            FileUtil().delete_directory(backup_path)
            FileUtil().zip_folder(self.model_folder_path, backup_path)
            logging.warning(f'Copied model from {self.model_folder_path} to {backup_path}')

        FileUtil().delete_directory(self.model_folder_path)

    def finish(self):
        if self._finished:
            return

        self._finished = True

        try:
            self.tensorboard.close()
        except Exception as err:
            logging.warning(f'Could not close tensorboard folder: {self.tensorboard_folder_path} - {err}')

        logging.warning(f'Finished training with epoch {self.epoch}')

        self._backup()
        HelperFuncs().close_logging()
        FileUtil().delete_directory(self.model_folder_path)

        if Configs().evaluation.remove_base:
            FileUtil().delete_file(StoragePath(f'{self.base_model_path}.zip'))

    def _execute_substages(self) -> bool:
        stage_changed = False

        substage = EpochInitStage(self)

        while substage:
            substage_completed: Optional[bool] = substage.run()
            DeviceDetector().empty_cache()

            if substage_completed is not None:
                stage_changed = (stage_changed or substage_completed)

            next_stage = substage.next_substage
            if next_stage is None:
                break
            substage = next_stage(self)

        return stage_changed

    @abstractmethod
    def run(self):
        pass
