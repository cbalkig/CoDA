import logging
import uuid
from typing import Optional

from configs.base.configs import Configs
from data.configuration.model import ModelConfiguration
from data.file.path import StoragePath
from data.types.operation_type import OperationType
from util.file_util import FileUtil


class Configuration:
    def __init__(
            self,
            model_folder_path: StoragePath,
            backup_folder_path: StoragePath,
            remote_folder_path: StoragePath,
            source_cross_val_k: Optional[int] = None,
            target_cross_val_k: Optional[int] = None,
            mlp_model: Optional[ModelConfiguration] = None,
            operation: OperationType = OperationType.TRAINING,
            number_of_classes: int = 0,
    ) -> None:
        self.model: ModelConfiguration = ModelConfiguration(model_folder_path, str(uuid.uuid4()))
        self.log_file_path: StoragePath = self.model.path.join(f"log.log")
        self.source_cross_val_k: Optional[int] = source_cross_val_k
        self.target_cross_val_k: Optional[int] = target_cross_val_k

        self.base_model: Optional[ModelConfiguration] = None

        if self.source_cross_val_k in Configs().pretrained.base_models.keys():
            base_model_uid = Configs().pretrained.base_models[self.source_cross_val_k]

            file_path = model_folder_path.join(f'{base_model_uid}.zip')
            if FileUtil().exists(file_path):
                self.base_model = ModelConfiguration(model_folder_path, base_model_uid)
            else:
                logging.warning(f"Base model does not exist: {file_path.path}")

                file_path = backup_folder_path.join(f'{base_model_uid}.zip')
                if FileUtil().exists(file_path):
                    self.base_model = ModelConfiguration(backup_folder_path, base_model_uid)
                else:
                    logging.warning(f"Base model does not exist: {file_path.path}")

                    file_path = remote_folder_path.join(f'{base_model_uid}.zip')
                    if FileUtil().exists(file_path):
                        self.base_model = ModelConfiguration(remote_folder_path, base_model_uid)
                    else:
                        logging.warning(f"Base model does not exist: {file_path.path}")
                        raise FileNotFoundError(f"File {base_model_uid} does not exist")

        self.mlp_model = mlp_model
        self.operation = operation
        self.number_of_classes: int = number_of_classes

        self.tensorboard_folder_path: StoragePath = self.model.path.join("tensorboard")
        FileUtil().create_directory(self.tensorboard_folder_path)

        logging.warning(f'Will log to --> {str(self.log_file_path)}...')
        logging.warning(f'Will log Tensorboard to --> {str(self.tensorboard_folder_path)}...')
