from configs.base.configs import Configs
from data.data_tag import DataTag
from data.dataset.data import Data
from data.model_tag import ModelTag
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.stage_types import Stages
from data.stages.stages.source.abstract_train_source import AbstractTrainSourceStage
from data.stages.stages.source.finetune_source_mixup import FinetuneSourceMixupStage
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from data.types.model_type import ModelType
from model.managers.dropout_manager import DropoutManager, DropoutType
from model.managers.mixup_manager import MixupManager
from model.model import Model


class FinetuneSourceDropoutStage(AbstractTrainSourceStage):
    def __init__(self) -> None:
        super().__init__(Stages.FINETUNE_SOURCE_DROPOUT, True, FinetuneSourceMixupStage)

        self.dropout_completed = None

    def _preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                    mixup_manager: MixupManager) -> None:
        model_tag = ModelTag(DataTag(DomainType.SOURCE, DataType.VALIDATION), DomainType.SOURCE)
        path = data.model_path.join(model_tag.best_model_tag)
        model.load(path)

        dropout_manager.add(epoch, DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR, DropoutType.DROPOUT,
                            initial_drop_out=Configs().feature_extractor.initial_drop_rate(self.stage),
                            peak_drop_out=Configs().feature_extractor.peak_drop_rate(self.stage),
                            delta=Configs().feature_extractor.dropout_delta(self.stage))

        dropout_manager.add(epoch, DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR, DropoutType.DROPOUT_PATH,
                            initial_drop_out=Configs().feature_extractor.initial_drop_path_rate(self.stage),
                            peak_drop_out=Configs().feature_extractor.peak_drop_path_rate(self.stage),
                            delta=Configs().feature_extractor.dropout_delta(self.stage))

        dropout_manager.add(epoch, DomainType.SOURCE, ModelType.CLASSIFIER, DropoutType.DROPOUT,
                            initial_drop_out=Configs().classifier.initial_drop_rate(self.stage),
                            peak_drop_out=Configs().classifier.peak_drop_rate(self.stage),
                            delta=Configs().classifier.dropout_delta(self.stage))

    def get_action_completed(self) -> bool:
        return super().get_action_completed() and self.dropout_completed

    def _init_epoch(self, init_stage: EpochInitStage) -> None:
        init_stage.dropout_manager.step(init_stage.epoch)

        init_stage.model.get_model(DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR).set_dropout(
            init_stage.dropout_manager.get(DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR, DropoutType.DROPOUT),
            init_stage.dropout_manager.get(DomainType.SOURCE, ModelType.FEATURE_EXTRACTOR, DropoutType.DROPOUT_PATH))

        init_stage.model.get_model(DomainType.SOURCE, ModelType.CLASSIFIER).set_dropout(
            init_stage.dropout_manager.get(DomainType.SOURCE, ModelType.CLASSIFIER, DropoutType.DROPOUT))

        self.dropout_completed = init_stage.dropout_manager.completed()
