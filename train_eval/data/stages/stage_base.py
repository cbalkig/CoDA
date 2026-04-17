from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Type, Dict, Tuple

from data.dataset.data import Data
from data.loss.loss import Loss
from data.model_tag import ModelTag
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.epoch_stages.train_stage import EpochTrainStage
from data.stages.stage_types import Stages
from data.stats.stats import Stats
from model.managers.dropout_manager import DropoutManager
from model.managers.mixup_manager import MixupManager
from model.managers.pseudo_label_manager import PseudoLabelManager
from model.model import Model

T = TypeVar('T', bound='StageBase')


class StageBase(ABC):
    stage: Stages

    def __init__(self, stage: Stages, continuous: bool = False, next_stage: Optional[Type['StageBase']] = None,
                 eval: bool = False) -> None:
        self._stage = stage
        self._continuous = continuous
        self._next_stage = next_stage
        self._eval = eval

    @property
    def stage(self) -> Stages:
        return self._stage

    @property
    def continuous(self) -> bool:
        return self._continuous

    @property
    def eval(self) -> bool:
        return self._eval

    @property
    def next_stage(self) -> Optional[Type['StageBase']]:
        return self._next_stage

    @abstractmethod
    def get_action_completed(self) -> bool:
        pass

    @abstractmethod
    def preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                   mixup_manager: MixupManager, pseudo_label_manager: PseudoLabelManager) -> None:
        pass

    @abstractmethod
    def run(self, train_stage: EpochTrainStage) -> None:
        pass

    @abstractmethod
    def init_epoch(self, init_stage: EpochInitStage) -> None:
        pass

    @abstractmethod
    def step_schedulers(self, model: Model, stats: Stats) -> None:
        pass

    @abstractmethod
    def get_improvement(self, losses: Dict[ModelTag, Loss], f1s: Dict[ModelTag, float]) -> Tuple[
        Dict[ModelTag, float], Dict[ModelTag, float]]:
        pass

    @abstractmethod
    def get_train_status(self) -> Dict[str, float | int]:
        pass
