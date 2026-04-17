from typing import Dict

from configs.base.configs import Configs
from data.data_tag import DataTag
from data.dataset.data import Data
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.stage_types import Stages
from data.stages.stages.source.abstract_train_source import AbstractTrainSourceStage
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from data.types.model_type import ModelType
from model.managers.dropout_manager import DropoutManager
from model.managers.mixup_manager import MixupManager
from model.managers.pseudo_label_manager import PseudoLabelManager
from model.model import Model


class TrainSourceStage(AbstractTrainSourceStage):
    def __init__(self) -> None:
        super().__init__(Stages.TRAIN_SOURCE, True, None)

    def _preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                    mixup_manager: MixupManager, pseudo_label_manager: PseudoLabelManager) -> None:
        data.sample_training([DataTag(DomainType.SOURCE, DataType.TRAIN)])
        data.sample_evaluation([DomainType.SOURCE])

        model.get_model(DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR).set_dropout(
            Configs().feature_extractor.drop_rate(self.stage), Configs().feature_extractor.drop_path_rate(self.stage))

        model.get_model(DomainType.SOURCE, ModelType.CLASSIFIER).set_dropout(
            Configs().classifier.drop_rate(self.stage))

        if not Configs().feature_extractor.unfrozen_all_layers:
            model.get_model(DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR).freeze_all()

    def _init_epoch(self, init_stage: EpochInitStage) -> None:
        pass

    def get_action_completed(self) -> bool:
        return super().get_action_completed()

    def get_train_status(self) -> Dict[str, float | int]:
        return {}
