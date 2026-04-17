import logging
import ssl
from typing import Tuple, Dict

from configs.base.configs import Configs
from data.configuration.configuration import Configuration
from data.dataset.data import Data
from data.file.path import StoragePath
from data.types.operation_type import OperationType
from managers.evaluation_manager import EvaluationManager
from managers.train_manager import TrainManager
from model.model import Model
from util.file_util import FileUtil
from util.helper_funcs import HelperFuncs


class Experiment:
    def __init__(self, config: Configuration):
        ssl._create_default_https_context = ssl._create_unverified_context

        self.config = config

    def get_data_model(self, source_folder_path: StoragePath, target_folder_path: StoragePath,
                       test_folder_paths: Dict[str, StoragePath]) -> Tuple[
        Model, Data]:
        FileUtil().create_directory(self.config.tensorboard_folder_path)

        data = Data(self.config.source_cross_val_k, self.config.target_cross_val_k, self.config.model.path,
                    source_folder_path, target_folder_path, test_folder_paths)
        model = Model(self.config.model.path, data.number_of_classes)

        logging.warning(f"Model ID: {self.config.model.model_id}")
        if self.config.base_model is not None:
            logging.warning(f"Base Model ID: {self.config.base_model.model_id}")

        if self.config.mlp_model is not None:
            logging.warning(f"MLP Model ID: {self.config.mlp_model.model_id}")

        return model, data

    def run(self, source_folder_path: StoragePath, target_folder_path: StoragePath,
            test_folder_paths: Dict[str, StoragePath]):
        FileUtil().create_directory(self.config.model.path)
        HelperFuncs().restart_logging(self.config.log_file_path)
        HelperFuncs().seed_everything()

        logging.warning(f'Config file path: {Configs().config_path}')
        logging.warning(f"Tag: {Configs().general.tag}")
        logging.warning(f"Source Cross Val K: {self.config.source_cross_val_k}")
        logging.warning(f"Target Cross Val K: {self.config.target_cross_val_k}")
        logging.warning(f"Operation: {self.config.operation.name}")
        logging.warning(f"Training Batch Size: {Configs().training.train_batch_size}")
        logging.warning(f"Evaluation Batch Size: {Configs().training.eval_batch_size}")

        cfg_file_path: StoragePath = Configs().config_path
        FileUtil().copy_file(cfg_file_path, self.config.model.path.join(cfg_file_path.name))

        model, data = self.get_data_model(source_folder_path, target_folder_path, test_folder_paths)
        if self.config.operation == OperationType.TRAINING:
            TrainManager(model, data, self.config).run()
        elif self.config.operation == OperationType.EVALUATION:
            EvaluationManager(model, data, self.config).run()
