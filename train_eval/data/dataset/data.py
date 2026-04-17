import logging
import os
from collections import defaultdict
from typing import Tuple, Sequence, Dict, List, Optional

import joblib
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from configs.base.configs import Configs
from data.data_tag import DataTag
from data.dataset.abstract import AbstractDataset
from data.dataset.file import FileDataset
from data.file.path import StoragePath
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from data.types.operation_type import OperationType
from util.file_util import FileUtil


class Data:
    def __init__(self, source_cross_val_k: Optional[int], target_cross_val_k: Optional[int], model_path: StoragePath,
                 source_folder_path: StoragePath, target_folder_path: StoragePath,
                 test_folder_paths: Dict[str, StoragePath]) -> None:
        self.source_cross_val_k = source_cross_val_k
        self.target_cross_val_k = target_cross_val_k
        self.model_path: StoragePath = model_path
        self.source_folder_path: StoragePath = source_folder_path
        self.target_folder_path: StoragePath = target_folder_path
        self.test_folder_paths: Dict[str, StoragePath] = test_folder_paths
        self.num_workers = Configs().dataset.num_workers

        logging.warning(f'Number of workers: {self.num_workers} of CPU count: {os.cpu_count()}')

        self.train_datasets: Dict[DataTag, List[StoragePath]] = {}
        self.eval_datasets: Dict[DataTag, List[StoragePath]] = {}

        self.data_sets: Dict[OperationType, Dict[DataTag, FileDataset]] = defaultdict(dict)
        self.data_loaders: Dict[
            OperationType, Dict[DataTag, Dict[str, DataLoader | torch.Tensor | Sequence[int]]]] = defaultdict(
            lambda: defaultdict(dict))

        self._label_encoder: Optional[LabelEncoder] = None

        self._initialize_datasets()

    @property
    def label_encoder(self) -> LabelEncoder:
        if self._label_encoder is None:
            encoder_path = self.model_path.join('label_encoder.joblib')
            if FileUtil().exists(encoder_path):
                self._label_encoder = joblib.load(encoder_path)
            else:
                source_data_tag = DataTag(DomainType.SOURCE, DataType.TRAIN)
                target_data_tag = DataTag(DomainType.TARGET, DataType.TRAIN)
                if source_data_tag in self.train_datasets:
                    labels = [path.parent_name for path in self.train_datasets[source_data_tag]]
                elif target_data_tag in self.train_datasets:
                    labels = [path.parent_name for path in self.train_datasets[target_data_tag]]
                else:
                    raise ValueError(f'Source and target is not set to the training data')

                self._label_encoder = LabelEncoder().fit(labels)
                FileUtil().dump(self._label_encoder, encoder_path)

        return self._label_encoder

    def _get_splits(self, cross_val_k: Optional[int], path: StoragePath, extensions: List[str]) -> Tuple[
        List[StoragePath], List[StoragePath], List[StoragePath]]:
        if path is None:
            return [], [], []

        logging.warning(f'Cross validation k: {cross_val_k}')
        if cross_val_k is not None:
            train_list = FileUtil().gather_files(path, f'train/k-fold-{cross_val_k}/**/*', extensions)
            val_list = FileUtil().gather_files(path, f'val/k-fold-{cross_val_k}/**/*', extensions)
        else:
            train_list = FileUtil().gather_files(path, f'train/**/*', extensions)
            val_list = FileUtil().gather_files(path, f'val*/**/*', extensions)

        if len(train_list) == 0 and len(val_list) == 0:
            train_list = FileUtil().gather_files(path, f'train/**/*', extensions)
            val_list = FileUtil().gather_files(path, f'val*/**/*', extensions)

        logging.warning(f'Number of train files before filter: {len(train_list)}')
        logging.warning(f'Number of val files before filter: {len(val_list)}')

        if Configs().dataset.dataset_type:
            train_list = Configs().dataset.dataset_type.filter(train_list)
            val_list = Configs().dataset.dataset_type.filter(val_list)

        logging.warning(f'Number of train files after filter: {len(train_list)}')
        logging.warning(f'Number of val files after filter: {len(val_list)}')

        test_list = FileUtil().gather_files(path, 'test/**/*', extensions)

        return train_list, val_list, test_list

    def _add_train_dataset(self, data_tag: DataTag, data_list: List[StoragePath]) -> None:
        if len(data_list) > 0:
            self.train_datasets[data_tag] = sorted(data_list, key=lambda path: str(path))

    def _add_evaluation_dataset(self, data_tag: DataTag, data_list: List[StoragePath]) -> None:
        if len(data_list) > 0:
            self.eval_datasets[data_tag] = sorted(data_list, key=lambda path: str(path))

    def _initialize_datasets(self) -> None:
        extensions = ['.jpg', '.jpeg', '.png']

        source_dir = self.source_folder_path
        target_dir = self.target_folder_path
        test_dirs = self.test_folder_paths

        logging.warning(f'Data augmentations: {Configs().dataset.data_augmentations}')

        logging.warning(f'Source directory: {source_dir}')
        logging.warning(f'Target directory: {target_dir}')

        for test_dir in test_dirs:
            logging.warning(f'Test directory: {test_dir} - {test_dirs[test_dir]}')

        source_train_list, source_val_list, source_test_list = self._get_splits(self.source_cross_val_k, source_dir,
                                                                                extensions)

        target_train_list, target_val_list, target_test_list = self._get_splits(self.target_cross_val_k, target_dir,
                                                                                extensions)

        test_lists: Dict[str, list[StoragePath]] = {}
        for test in test_dirs:
            test_dir = test_dirs[test]
            test_lists[test] = FileUtil().gather_files(test_dir, '**/*', extensions) if test_dir else []
            logging.warning(f'Number of test samples: {test} - {len(test_lists[test])}')

        logging.warning(f'Number of source training samples: {len(source_train_list)}')
        logging.warning(f'Number of source validation samples: {len(source_val_list)}')
        logging.warning(f'Number of source test samples: {len(source_test_list)}')

        logging.warning(f'Number of target training samples: {len(target_train_list)}')
        logging.warning(f'Number of target validation samples: {len(target_val_list)}')
        logging.warning(f'Number of target test samples: {len(target_test_list)}')

        self._add_train_dataset(data_tag=DataTag(DomainType.SOURCE, DataType.TRAIN), data_list=source_train_list)
        self._add_evaluation_dataset(data_tag=DataTag(DomainType.SOURCE, DataType.TRAIN), data_list=source_train_list)
        self._add_evaluation_dataset(data_tag=DataTag(DomainType.SOURCE, DataType.VALIDATION),
                                     data_list=source_val_list)
        self._add_evaluation_dataset(data_tag=DataTag(DomainType.SOURCE, DataType.TEST), data_list=source_test_list)

        self._add_train_dataset(data_tag=DataTag(DomainType.TARGET, DataType.TRAIN), data_list=target_train_list)
        self._add_evaluation_dataset(data_tag=DataTag(DomainType.TARGET, DataType.TRAIN), data_list=target_train_list)
        self._add_evaluation_dataset(data_tag=DataTag(DomainType.TARGET, DataType.VALIDATION),
                                     data_list=target_val_list)
        self._add_evaluation_dataset(data_tag=DataTag(DomainType.TARGET, DataType.TEST), data_list=target_test_list)

        for test in test_dirs:
            self._add_evaluation_dataset(data_tag=DataTag(DomainType.SOURCE, DataType.TEST, identifier=test),
                                         data_list=test_lists[test])

            self._add_evaluation_dataset(data_tag=DataTag(DomainType.TARGET, DataType.TEST, identifier=test),
                                         data_list=test_lists[test])

        self.number_of_classes = len(self.label_encoder.classes_)

    def get_batch(self, dataset: AbstractDataset, is_train: bool) -> Tuple[DataLoader, torch.Tensor, Sequence[int]]:
        sampler, class_weights, samples_per_classes = dataset.get_sampler()

        batch_size = Configs().training.train_batch_size if is_train else Configs().training.eval_batch_size

        use_sampler = bool(is_train and sampler is not None)
        loader_kwargs = {
            'shuffle': False if use_sampler else is_train,
            'batch_size': batch_size,
            'num_workers': self.num_workers,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': False,
            'sampler': sampler if use_sampler else None,
            'prefetch_factor': 1
        }

        if self.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 1

        loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

        return DataLoader(dataset, **loader_kwargs), class_weights, samples_per_classes

    def sample_evaluation(self, domain_types: List[Optional[DomainType]]) -> None:
        if domain_types is not None:
            data_tags = self.get_evaluation_tags(domain_types)

            for data_tag in data_tags:
                self.data_sets[OperationType.EVALUATION][data_tag] = FileDataset(
                    data_augment=False, label_encoder=self.label_encoder,
                    num_classes=len(self.label_encoder.classes_),
                    paths=self.eval_datasets[data_tag])
        else:
            for data_tag in self.eval_datasets.keys():
                self.data_sets[OperationType.EVALUATION][data_tag] = FileDataset(
                    data_augment=False, label_encoder=self.label_encoder,
                    num_classes=len(self.label_encoder.classes_),
                    paths=self.eval_datasets[data_tag])

        self.set_data_loaders()

    def sample_training(self, data_tags: List[DataTag]) -> None:
        for data_tag in data_tags:
            self.data_sets[OperationType.TRAINING][data_tag] = FileDataset(
                data_augment=True, label_encoder=self.label_encoder,
                num_classes=len(self.label_encoder.classes_),
                paths=self.train_datasets[data_tag])

        self.set_data_loaders()

    def set_data_loaders(self):
        self.reset_dataloaders()

        for operation, items in self.data_sets.items():
            for data_tag, dataset in list(items.items()):
                if len(dataset) == 0:
                    if operation in self.data_loaders and data_tag in self.data_loaders[operation]:
                        del self.data_loaders[operation][data_tag]
                else:
                    loader, class_weights, samples_per_classes = self.get_batch(dataset,
                                                                                is_train=operation == OperationType.TRAINING)

                    self.data_loaders[operation][data_tag] = {
                        'loader': loader,
                        'class_weights': class_weights,
                        'samples_per_classes': samples_per_classes
                    }

    def get_train_dataloaders(self, data_tag: DataTag) -> Tuple[DataLoader, torch.Tensor, Sequence[int]] | None:
        data = self.data_loaders.get(OperationType.TRAINING, {}).get(data_tag)
        if data is None:
            return None

        return data['loader'], data['class_weights'], data['samples_per_classes']

    def get_evaluation_dataloaders(self, data_tag: DataTag) -> Tuple[
                                                                   DataLoader, torch.Tensor,
                                                                   Sequence[int]] | None:
        data = self.data_loaders.get(OperationType.EVALUATION, {}).get(data_tag)
        if data:
            return data['loader'], data['class_weights'], data['samples_per_classes']

        return None

    def get_evaluation_tags(self, domain_types: Optional[List[DomainType]] = None) -> List[DataTag]:
        result: List[DataTag] = []

        for data_tag in self.eval_datasets.keys():
            if domain_types is not None:
                if data_tag.domain in domain_types:
                    result.append(data_tag)
            else:
                result.append(data_tag)

        return result

    def reset_dataloaders(self) -> None:
        keys = list(self.data_loaders.keys())
        for loader in keys:
            del self.data_loaders[loader]
