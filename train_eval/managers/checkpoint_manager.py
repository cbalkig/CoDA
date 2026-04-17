import logging
import random
from typing import Optional

import numpy as np
import torch

from configs.base.configs import Configs
from data.file.path import StoragePath
from data.stages.stage_history import StageHistory
from data.stats.stats import Stats
from managers.nns import NetworkStabilityScheduler
from util.file_util import FileUtil


class CheckpointManager:
    def __init__(self, log_file_path: StoragePath, tensorboard_folder_path: StoragePath, manager: 'TrainManager',
                 data: 'Data'):
        self.log_file_path: StoragePath = log_file_path
        self.tensorboard_folder_path: StoragePath = tensorboard_folder_path
        self.data = data
        self.manager = manager

    def save(self, epoch: int, sub_folder="checkpoint") -> None:
        folder: StoragePath = self.manager.model_folder_path.join(sub_folder)
        FileUtil().create_directory(folder)

        FileUtil().copy_folder(self.tensorboard_folder_path, folder.join('logs'))
        self.manager.stage_manager.history.save(folder.join('stage_history.pt'))
        self.manager.stage_manager.stage_scheduler.save(folder.join('stage_scheduler.pt'))
        self.manager.stats.save(folder.join('stats.pt'))

        epoch_data = {'epoch': epoch}
        FileUtil().dump(epoch_data, folder.join('epoch_data.pkl'))

        general_data = {
            'source k-fold': Configs().dataset.source_cross_val_k,
            'target k-fold': Configs().dataset.target_cross_val_k,
        }
        FileUtil().dump(general_data, folder.join('general_data.pkl'))
        FileUtil().dump(self.data.label_encoder, folder.join('label_encoder.pkl'))
        FileUtil().dump(self.manager.mixup_manager, folder.join('mixup_manager.pkl'))
        FileUtil().dump(self.manager.dropout_manager, folder.join('dropout_manager.pkl'))

        self.manager.model.save(folder)

        random_state_dict = {
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'mps_rng_state': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        }

        FileUtil().dump(random_state_dict, folder.join('random.pt'))
        FileUtil().copy_file(self.log_file_path, self.manager.model_folder_path.join('log.log'))

        logging.debug(f'Checkpoint saved at epoch: {epoch}')

    def load(self, sub_folder: Optional[str] = None, folder: Optional[StoragePath] = None) -> None:
        if folder is None:
            folder = self.manager.model_folder_path

        if sub_folder is None:
            sub_folder = "checkpoint"

        model_folder: StoragePath = folder.join(sub_folder)
        logging.info(f'Resuming from checkpoint: {model_folder}')

        random_checkpoint = FileUtil().load(model_folder.join('random.pt'), weights_only=False)

        torch.set_rng_state(random_checkpoint['torch_rng_state'].cpu())
        np.random.set_state(random_checkpoint['numpy_rng_state'])
        random.setstate(random_checkpoint['python_rng_state'])

        if torch.cuda.is_available() and random_checkpoint.get('cuda_rng_state'):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in random_checkpoint['cuda_rng_state']]
            )

        if torch.backends.mps.is_available():
            mps_state = random_checkpoint.get("mps_rng_state", None)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state.cpu())

        self.data._label_encoder = FileUtil().load(model_folder.join('label_encoder.pkl'))
        self.manager.model.load(model_folder, Configs().pretrained.weights_only)

        epoch_data = FileUtil().load(model_folder.join('epoch_data.pkl'))
        epoch = epoch_data.get('epoch', 0)

        if not Configs().pretrained.weights_only:
            self.manager.stage_manager.history = StageHistory.load(model_folder.join('stage_history.pt'))
            self.manager.stage_manager.stage_scheduler = NetworkStabilityScheduler.load(
                model_folder.join('stage_scheduler.pt'))
            self.manager.mixup_manager = FileUtil().load(model_folder.join('mixup_manager.pkl'))
            self.manager.dropout_manager = FileUtil().load(model_folder.join('dropout_manager.pkl'))

            self.manager.stats = Stats.load(model_folder.join('stats.pt'))

            log_folder = (
                model_folder.join('logs') if sub_folder == 'checkpoint' else self.tensorboard_folder_path
            )
            FileUtil().copy_folder(log_folder, self.tensorboard_folder_path)
            self.manager.initial_epoch = epoch
        else:
            FileUtil().delete_directory(self.tensorboard_folder_path)

        FileUtil().copy_files_with_extensions(folder, self.manager.model_folder_path.join('previous_data'),
                                              extensions=['yaml', 'log'])

        FileUtil().copy_folders_with_prefix(folder, self.manager.model_folder_path,
                                            prefixes=['best_', 'checkpoint'])

        logging.warning(f'Resumed from epoch: {epoch}')
