import logging
from abc import abstractmethod
from typing import Optional, Type, List, Dict, Tuple

from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.base.configs import Configs
from data.data_tag import DataTag
from data.dataset.data import Data
from data.domain_losses import DomainLosses
from data.loss.loss import Loss
from data.loss.losses import Losses
from data.model_tag import ModelTag
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.epoch_stages.train_stage import EpochTrainStage
from data.stages.stage_base import StageBase
from data.stages.stage_status import StageStatus
from data.stages.stage_types import Stages
from data.stats.stats import Stats
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from data.types.model_type import ModelType
from model.managers.dropout_manager import DropoutManager, DropoutType
from model.managers.mixup_manager import MixupManager
from model.managers.pseudo_label_manager import PseudoLabelManager
from model.model import Model
from model.submodel import SubModel


class AbstractTrainStage(StageBase):
    _CLIP_MAX_NORM = 1.0

    def __init__(self, domain: DomainType, stage: Stages, continuous: bool = False,
                 next_stage: Optional[Type['StageBase']] = None):
        super().__init__(stage, continuous, next_stage)
        self.domain = domain

        self.max_unfreeze_layers = None
        self.current_unfrozen_layers = None

        self.trained: bool = False
        self.resume: bool = False

    @abstractmethod
    def _preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                    mixup_manager: MixupManager, pseudo_label_manager: PseudoLabelManager) -> None:
        pass

    @abstractmethod
    def _init_epoch(self, init_stage: EpochInitStage) -> None:
        pass

    def preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                   mixup_manager: MixupManager, pseudo_label_manager: PseudoLabelManager) -> Data:
        data.reset_dataloaders()
        dropout_manager.reset()
        mixup_manager.reset()
        pseudo_label_manager.reset()

        self.resume = False
        for domain in DomainType:
            if domain == self.domain:
                if domain not in model.keys:
                    model.add_model(self.domain, SubModel(model.number_of_classes, dropouts={
                        ModelType.FEATURE_EXTRACTOR: {
                            DropoutType.DROPOUT: Configs().feature_extractor.drop_rate(self.stage),
                            DropoutType.DROPOUT_PATH: Configs().feature_extractor.drop_path_rate(self.stage)
                        },
                        ModelType.CLASSIFIER: {
                            DropoutType.DROPOUT: Configs().classifier.drop_rate(self.stage)
                        },
                    }, learning_rates={
                        ModelType.FEATURE_EXTRACTOR: Configs().feature_extractor.optimizer_learning_rate(self.stage),
                        ModelType.CLASSIFIER: Configs().classifier.optimizer_learning_rate(self.stage)
                    }))
                    logging.warning(f'Model created for domain: {self.domain}')
                else:
                    self.resume = True
            else:
                model.remove_models(domain)
                logging.warning(f'Removed {domain} models')

        model.get_models(self.domain).set_learning_rate(ModelType.FEATURE_EXTRACTOR,
                                                        Configs().feature_extractor.optimizer_learning_rate(
                                                            self.stage))
        model.get_models(self.domain).set_learning_rate(ModelType.CLASSIFIER,
                                                        Configs().classifier.optimizer_learning_rate(self.stage))

        self._preprocess(epoch, model, data, dropout_manager, mixup_manager, pseudo_label_manager)

        return data

    def init_epoch(self, init_stage: EpochInitStage) -> None:
        fe_model = init_stage.model.get_model(self.domain, ModelType.FEATURE_EXTRACTOR)

        if init_stage.stage_scheduler.stage_status in [StageStatus.PLATEAU, StageStatus.DIVERGING,
                                                       StageStatus.FLUCTUATING]:
            one_more = fe_model.number_of_unfrozen_layers + 1
            if one_more <= self.max_unfreeze_layers:
                changed = fe_model.unfreeze_last_n_layers(one_more)
                if changed:
                    init_stage.stage_scheduler.reset()

        self.max_unfreeze_layers = fe_model.number_of_layers
        self.current_unfrozen_layers = fe_model.number_of_unfrozen_layers

        self._init_epoch(init_stage)

    @staticmethod
    def _get_improvement(model_tags: List[ModelTag], losses: Dict[ModelTag, Loss], f1s: Dict[ModelTag, float]) -> \
            Tuple[
                Dict[ModelTag, float], Dict[ModelTag, float]]:
        _losses: Dict[ModelTag, float] = {}
        _f1s: Dict[ModelTag, float] = {}

        for model_tag in model_tags:
            _losses[model_tag] = losses[model_tag].get_average()
            _f1s[model_tag] = f1s[model_tag]

        return _losses, _f1s

    @abstractmethod
    def get_action_completed(self) -> bool:
        if self.current_unfrozen_layers is None:
            return False

        return self.current_unfrozen_layers >= self.max_unfreeze_layers

    def step_schedulers(self, model: Model, stats: Stats) -> None:
        if not self.trained:
            return

        eval_stats = stats.get_last_eval_stats(
            ModelTag(DataTag(self.domain, DataType.VALIDATION), self.domain))

        if eval_stats is None:
            eval_stats = stats.get_last_eval_stats(
                ModelTag(DataTag(self.domain, DataType.TRAIN), self.domain))

        if Configs().feature_extractor.scheduler_type == CosineAnnealingLR:
            model.get_scheduler(self.domain, ModelType.FEATURE_EXTRACTOR).step()
        else:
            model.get_scheduler(self.domain, ModelType.FEATURE_EXTRACTOR).step(eval_stats.metrics.f1)

        if Configs().classifier.scheduler_type == CosineAnnealingLR:
            model.get_scheduler(self.domain, ModelType.CLASSIFIER).step()
        else:
            model.get_scheduler(self.domain, ModelType.CLASSIFIER).step(eval_stats.metrics.f1)

    def get_total_loss(self, stage: EpochTrainStage) -> Losses:
        return Losses({
            self.domain: DomainLosses(stage.lambdas)
        })
