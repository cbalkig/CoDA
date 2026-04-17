import logging
from typing import Type, List

import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, \
    precision_score

from data.loss.focal_loss import ClassBalancedFocalLoss
from data.loss.loss import Loss
from data.metrics import Metrics
from data.model_tag import ModelTag
from data.stages.epoch_stages.base_stage import EpochStageBase
from data.stages.epoch_stages.update_stage import EpochUpdateStage
from data.stats.epoch_eval_stats import EpochEvalStats
from data.types.model_type import ModelType
from util.device_detector import DeviceDetector


class EpochEvalStage(EpochStageBase):
    def __init__(self, manager: 'TrainManager'):
        self.model: 'Model' = manager.model
        self.epoch: int = manager.epoch
        self.data: 'Data' = manager.data
        self.stage: 'Stage' = manager.stage_manager.get_current_stage()
        self.stage_type: 'Stages' = manager.stage_manager.get_current_stage_type()
        self.stats: 'Stats' = manager.stats

    def _generate_eval_stats(self, loss: Loss, labels: List[int], preds: List[int]) -> EpochEvalStats:
        metrics = Metrics(
            accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0),
            precision_score(labels, preds, average="macro", zero_division=0),
            recall_score(labels, preds, average="macro", zero_division=0),
            classification_report(labels, preds, output_dict=True, zero_division=0),
            confusion_matrix(labels, preds).tolist()
        )
        return EpochEvalStats(self.epoch, loss, labels, preds, metrics)

    def _evaluate(self, model_tag: ModelTag) -> None:
        labels, preds = [], []

        result = self.data.get_evaluation_dataloaders(model_tag.data_tag)
        if result is None:
            return None

        loader, _, sequence_per_classes = result
        if loader is None:
            return None

        loss = Loss()
        criterion = ClassBalancedFocalLoss(sequence_per_classes)

        for batch in loader:
            images, batch_labels = DeviceDetector().to(batch[1]), DeviceDetector().to(batch[2])
            labels.extend(batch_labels.cpu().tolist())

            with torch.no_grad():
                features = self.model.get_model(model_tag.eval_on, ModelType.FEATURE_EXTRACTOR)(images)
                logits = self.model.get_model(model_tag.eval_on, ModelType.CLASSIFIER)(features.detach())

                loss_value = criterion(logits, batch_labels)

                preds.extend(logits.argmax(dim=1).cpu().tolist())

                loss.add(Loss(value=loss_value, item_count=images.size(0)))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.mps.is_available():
            torch.mps.synchronize()

        stats = self._generate_eval_stats(loss, labels, preds)
        self.stats.add_evaluation_stats(model_tag, stats)

        logging.info(
            f'Epoch: {self.epoch} - Evaluation stats - {model_tag.tag} - F1: {(stats.metrics.f1 * 100):.2f}%')

    def run(self) -> None:
        if self.stage.stage.eval:
            return

        self.model.eval_mode()

        for item in self.data.get_evaluation_tags([self.stage.stage.domain]):
            self._evaluate(ModelTag(item, self.stage.stage.domain))

    @property
    def next_substage(self) -> Type[EpochStageBase] | None:
        return EpochUpdateStage
