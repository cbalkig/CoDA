import argparse
from pathlib import Path

import yaml

from configs.base.config import Config
from configs.classifier import ClassifierConfig
from configs.data_set import DatasetConfig
from configs.evaluation import EvaluationConfig
from configs.feature_extractor import FeatureExtractorConfig
from configs.focal_loss import FocalLossConfig
from configs.general import GeneralConfig
from configs.mixup_criterion import MixupCriterionConfig
from configs.pretrained import PretrainedConfig
from configs.pseudo_label import PseudoLabelConfig
from configs.stage_scheduler import StageSchedulerConfig
from configs.storage import StorageConfig
from configs.training import TrainingConfig
from data.file.path import StoragePath


class Configs:
    _initialized = False
    _cache = None

    def __init__(self):
        if self.__class__._initialized and self.__class__._cache is not None:
            self.__dict__ = self.__class__._cache.__dict__.copy()
            return

        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg_file", type=str, default=None, help="Path to config YAML file")
        args = parser.parse_args()
        if args.cfg_file:
            path = args.cfg_file
        else:
            path = "main.yaml"

        self._path = Path(path)
        with open(self._path, "r") as f:
            data = yaml.safe_load(f) or {}

        _cfg = Config(data)

        self._general = GeneralConfig(_cfg["general"])
        self._storage = StorageConfig(_cfg["storage"])
        self.pretrained = PretrainedConfig(_cfg["pretrained"])
        self.dataset = DatasetConfig(_cfg["dataset"])
        self.training = TrainingConfig(_cfg["training"])
        self.feature_extractor = FeatureExtractorConfig(_cfg["feature_extractor"])
        self.classifier = ClassifierConfig(_cfg["classifier"])
        self.pseudo_label = PseudoLabelConfig(_cfg["pseudo_label"])
        self.focal_loss = FocalLossConfig(_cfg["focal_loss"])
        self.mixup_criterion = MixupCriterionConfig(_cfg["mixup_criterion"])
        self.stage_scheduler = StageSchedulerConfig(_cfg["stage_scheduler"])
        self.evaluation = EvaluationConfig(_cfg["evaluation"])

        # cache and mark initialized
        self.__class__._cache = self
        self.__class__._initialized = True

    @property
    def config_path(self) -> StoragePath:
        return StoragePath(self._path.absolute())

    @property
    def general(self) -> GeneralConfig:
        return self._general

    @property
    def storage(self) -> StorageConfig:
        return self._storage
