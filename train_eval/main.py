import logging
import os
import shutil

from data.stages.stage_types import Stages

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import Optional, Dict, List

from configs.base.configs import Configs
from data.configuration.configuration import Configuration
from data.experiment import Experiment
from data.file.path import StoragePath
from data.types.operation_type import OperationType
from util.file_util import FileUtil
from util.helper_funcs import HelperFuncs


def download_directory():
    remote_folder: StoragePath = Configs().storage.remote_folder
    local_folder: StoragePath = Configs().storage.local_folder

    if local_folder.local:
        local_folder.path.mkdir(parents=True, exist_ok=True)

    local_state_file: StoragePath = local_folder.join("local_state.pkl")

    previous_remote_source: Optional[StoragePath] = None
    if FileUtil().exists(local_state_file):
        data = FileUtil().read_pickle_file(local_state_file)
        previous_remote_source = StoragePath(data.get('remote_source'))

    if previous_remote_source != remote_folder:
        logging.warning(f"Downloading from {remote_folder} to {local_folder}...")

        if local_folder.path == remote_folder.path:
            logging.warning(f'Same directory as {local_folder} and {remote_folder}. Skipping.')
        else:
            if FileUtil().exists(local_folder):
                FileUtil().delete_directory(local_folder)

            if remote_folder.name.endswith(".zip"):
                FileUtil().unzip_file(remote_folder, local_folder)
            else:
                FileUtil().download_directory(remote_folder, local_folder.path)

        FileUtil().write_pickle_file(local_state_file, {'remote_source': remote_folder.path})
    else:
        logging.info("Local directory is up to date.")


if __name__ == "__main__":
    HelperFuncs().setup_logging()

    if Configs().storage.download:
        download_directory()

    models_folder_path: StoragePath = Configs().storage.destination_folder.join('Models')
    source_folder_path: StoragePath = Configs().storage.source_folder
    target_folder_path: StoragePath = Configs().storage.target_folder
    remote_folder_path: StoragePath = Configs().storage.remote_folder
    test_folder_paths: Dict[str, StoragePath] = Configs().storage.test_folders
    backup_folder_path: StoragePath = Configs().storage.backup_folder

    operation: OperationType = Configs().general.operation

    if operation == OperationType.EVALUATION:
        if Configs().evaluation.report_folder is None:
            raise RuntimeError('Evaluation report folder not set')

        if Configs().evaluation.cleanup and Configs().evaluation.report_folder.exists():
            shutil.rmtree(Configs().evaluation.report_folder)

        Configs().evaluation.report_folder.mkdir(parents=True, exist_ok=True)


    def _run_experiment(source_cross_val_id: Optional[int] = None, target_cross_val_id: Optional[int] = None) -> None:
        logging.warning(f"Running experiment Source CV: {source_cross_val_id} - Target CV: {target_cross_val_id}")

        try:
            experiment_config = Configuration(
                source_cross_val_k=source_cross_val_id,
                target_cross_val_k=target_cross_val_id,
                operation=operation,
                model_folder_path=models_folder_path,
                backup_folder_path=backup_folder_path,
                remote_folder_path=remote_folder_path,
            )
        except Exception as err:
            logging.error(f'Error on configuration load: {err}')
            return None

        experiment = Experiment(experiment_config)
        experiment.run(source_folder_path, target_folder_path, test_folder_paths)


    experiment_config = None
    source_cross_val_k: None | List[int] = Configs().dataset.source_cross_val_k
    target_cross_val_k: None | List[int] = Configs().dataset.target_cross_val_k
    if source_cross_val_k is None:
        if Configs().general.stage in [Stages.TRAIN_TARGET] or Configs().general.operation == OperationType.EVALUATION:
            if target_cross_val_k is None:
                _run_experiment(source_cross_val_k, target_cross_val_k)
            else:
                for target_cross_val_idx in target_cross_val_k:
                    _run_experiment(source_cross_val_k, target_cross_val_idx)
        elif Configs().general.stage is None or Configs().general.stage == Stages.TRAIN_SOURCE:
            _run_experiment(source_cross_val_k, None)
        else:
            raise NotImplementedError(f'Unknown stage: {Configs().general.stage}')
    else:
        for source_cross_val_idx in source_cross_val_k:
            if Configs().general.stage in [Stages.TRAIN_TARGET, Stages.TRAIN_UPPER_TARGET] or Configs().general.operation == OperationType.EVALUATION:
                if target_cross_val_k is None:
                    _run_experiment(source_cross_val_idx, target_cross_val_k)
                else:
                    _run_experiment(source_cross_val_idx, source_cross_val_idx)
            elif Configs().general.stage == Stages.TRAIN_SOURCE:
                _run_experiment(source_cross_val_idx, None)
            else:
                raise NotImplementedError(f'Unknown stage: {Configs().general.stage}')
