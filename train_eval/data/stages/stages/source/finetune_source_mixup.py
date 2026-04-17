from configs.base.configs import Configs
from data.data_tag import DataTag
from data.dataset.data import Data
from data.model_tag import ModelTag
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.stage_types import Stages
from data.stages.stages.eval.eval_model import EvaluateModelTypeStage
from data.stages.stages.source.abstract_train_source import AbstractTrainSourceStage
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from model.managers.dropout_manager import DropoutManager
from model.managers.mixup_manager import MixupManager
from model.model import Model


class FinetuneSourceMixupStage(AbstractTrainSourceStage):
    def __init__(self) -> None:
        super().__init__(Stages.FINETUNE_SOURCE_MIXUP, True, EvaluateModelTypeStage)

        self.mixup_prob_completed = None

    def _preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                    mixup_manager: MixupManager) -> None:
        model_tag = ModelTag(DataTag(DomainType.SOURCE, DataType.VALIDATION), DomainType.SOURCE)
        path = data.model_path.join(model_tag.best_model_tag)
        model.load(path)

        mixup_manager.add(epoch, Configs().mixup_criterion.mixup_prob(self.stage),
                          Configs().mixup_criterion.scheduler_mode(self.stage),
                          Configs().mixup_criterion.mixup_prob_delta(self.stage), )

    def get_action_completed(self) -> bool:
        return super().get_action_completed() and self.mixup_prob_completed

    def _init_epoch(self, init_stage: EpochInitStage) -> None:
        init_stage.mixup_manager.step(init_stage.epoch)

        self.mixup_prob_completed = init_stage.mixup_manager.completed()
