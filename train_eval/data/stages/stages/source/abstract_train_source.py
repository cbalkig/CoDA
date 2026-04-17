import logging
from typing import Optional, Type, Dict, Tuple

import torch
import torch.nn.utils as nn_utils

from configs.base.configs import Configs
from data.data_tag import DataTag
from data.loss.loss import Loss
from data.loss.mixup_loss import MixupCriterion
from data.model_tag import ModelTag
from data.stages.epoch_stages.train_stage import EpochTrainStage
from data.stages.stage_base import StageBase
from data.stages.stage_types import Stages
from data.stages.stages.abstract_train import AbstractTrainStage
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from data.types.model_type import ModelType
from util.device_detector import DeviceDetector


class AbstractTrainSourceStage(AbstractTrainStage):
    def __init__(self, stage: Stages, continuous: bool, next_stage: Optional[Type[StageBase]]) -> None:
        super().__init__(DomainType.SOURCE, stage, continuous, next_stage)

    def get_improvement(self, losses: Dict[ModelTag, Loss], f1s: Dict[ModelTag, float]) -> Tuple[
        Dict[ModelTag, float], Dict[ModelTag, float]]:
        return self._get_improvement([ModelTag(DataTag(DomainType.SOURCE, DataType.TRAIN), DomainType.SOURCE),
                                      ModelTag(DataTag(DomainType.SOURCE, DataType.VALIDATION), DomainType.SOURCE)],
                                     losses,
                                     f1s)

    def run(self, train_stage: EpochTrainStage) -> None:
        train_stage.model.get_models(self.domain).train()

        data = train_stage.data.get_train_dataloaders(
            DataTag(self.domain, DataType.TRAIN))

        fe = train_stage.model.get_model(self.domain, ModelType.FEATURE_EXTRACTOR)
        mlp = train_stage.model.get_model(self.domain, ModelType.CLASSIFIER)

        if data is not None:
            self.trained = True
            loader, class_weights, samples_per_classes = data
            logging.info(f"Data loaded: {len(loader.dataset)} samples")
            logging.info(f'Samples per class: {samples_per_classes}')
            logging.info(f'Class weights: {class_weights}')

            if len(loader.dataset) < 1:
                self.trained = False
                train_stage.total_loss.losses[self.domain].classification.add(Loss(value=torch.tensor(0), item_count=0))

            criterion = MixupCriterion(train_stage.data.number_of_classes, class_weights, samples_per_classes,
                                       train_stage.mixup_manager.mixup_prob,
                                       mode=Configs().mixup_criterion.mode(self.stage),
                                       cutmix_alpha=Configs().mixup_criterion.cutmix_alpha(self.stage),
                                       mixup_alpha=Configs().mixup_criterion.mixup_alpha(self.stage),
                                       label_smoothing=Configs().mixup_criterion.label_smoothing(self.stage),
                                       switch_prob=Configs().mixup_criterion.switch_prob(self.stage))

            optimizer_fe = train_stage.model.get_optimizer(self.domain, ModelType.FEATURE_EXTRACTOR)
            optimizer_mlp = train_stage.model.get_optimizer(self.domain, ModelType.CLASSIFIER)

            for batch in loader:
                _, source_images, source_labels, _ = batch

                if len(source_images) < 2:
                    continue

                source_images = DeviceDetector().to(source_images)
                source_labels = DeviceDetector().to(source_labels)

                source_images, source_labels = criterion.prepare_batch(source_images, source_labels)

                optimizer_fe.zero_grad(set_to_none=True)
                optimizer_mlp.zero_grad(set_to_none=True)

                features = fe(source_images)
                logits = mlp(features)

                loss_value = criterion(logits, source_labels)
                final_loss = loss_value * train_stage.lambdas.in_domain.classification

                final_loss.backward()

                nn_utils.clip_grad_norm_(fe.parameters(), self._CLIP_MAX_NORM)
                nn_utils.clip_grad_norm_(mlp.parameters(), self._CLIP_MAX_NORM)

                optimizer_fe.step()
                optimizer_mlp.step()

                logging.debug(f"Source Loss: {final_loss.item():.10f}")

                train_stage.total_loss.losses[self.domain].classification.add(
                    Loss(value=final_loss, item_count=source_images.size(0)))
        else:
            self.trained = False
            train_stage.total_loss.losses[self.domain].classification.add(
                Loss(value=torch.tensor(0), item_count=0))

        DeviceDetector().empty_cache()

        train_stage.train_epoch_status.source_num_unfrozen_layers = fe.number_of_unfrozen_layers
        train_stage.train_epoch_status.source_num_layers = fe.number_of_layers
