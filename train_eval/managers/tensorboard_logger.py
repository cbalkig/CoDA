import atexit
import threading
from typing import Dict

import math
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

from data.dataset.data import Data
from data.domain_losses import DomainLosses
from data.file.path import Path, StoragePath
from data.indomain_lambdas import InDomainLambdas
from data.lambdas import Lambdas
from data.loss.loss import Loss
from data.loss.losses import Losses
from data.model_tag import ModelTag
from data.stats.epoch_stage_status import EpochStageStatus
from data.stats.epoch_train_stats import EpochTrainStats
from data.stats.epoch_train_status import EpochTrainStatus
from data.stats.stats import Stats
from data.types.domain_type import DomainType

# ---------- Shared writer cache ----------
_writer_lock = threading.Lock()
_writer_cache: Dict[str, SummaryWriter] = {}


def _get_or_create_writer(log_dir: Path) -> SummaryWriter:
    key = str(log_dir)
    with _writer_lock:
        if key not in _writer_cache:
            _writer_cache[key] = SummaryWriter(log_dir=key)
        return _writer_cache[key]


def _flush_all_writers():
    with _writer_lock:
        for w in _writer_cache.values():
            try:
                w.flush()
            except Exception:
                pass


atexit.register(_flush_all_writers)


# ---------- Safe logging helpers ----------
def _to_float_or_none(x):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _clean_scalars(d: Dict[str, object]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        v = _to_float_or_none(v)
        if v is not None:
            out[k] = v
    return out


class TensorboardLogger:
    def __init__(self, log_dir: StoragePath):
        self.tensorboard_process = None
        self.log_dir = log_dir
        self.local_writer = _get_or_create_writer(self.log_dir.path)

    def log_epoch_data(self, epoch: int, stats: Stats, data: Data):
        train_stats = stats.get_last_train_stats()

        eval_losses: Dict[ModelTag, Loss] = {}
        for data_tag in data.get_evaluation_tags():
            for domain in DomainType:
                model_tag = ModelTag(data_tag, domain)
                result = stats.get_last_eval_stats(model_tag)
                if result is not None:
                    eval_losses[model_tag] = result.loss

        self._log_loss_components(epoch, train_stats, eval_losses)
        self._log_status(epoch, train_stats.status)
        self._log_stage_status(epoch, train_stats.stage_status)
        self._log_lambdas(epoch, train_stats.status.lambdas)
        self._log_learning_rates(epoch, train_stats)
        self._log_metrics(epoch, stats, data.label_encoder)

        self.local_writer.flush()

    def _log_loss_components(self, epoch: int, train_stats: EpochTrainStats,
                             eval_losses: Dict[ModelTag, Loss]):
        self._log_total_loss(epoch, train_stats.losses)
        self._log_weighted_losses(epoch, train_stats.losses, train_stats.status.lambdas)
        self._log_loss_type(epoch, train_stats.losses)
        self._log_eval_losses(epoch, eval_losses)

    def _log_total_loss(self, epoch: int, losses: Losses):
        # losses.losses: Dict[DomainType, DomainLosses]
        data = _clean_scalars({
            str(domain): domain_losses.get_weighted_total()
            for domain, domain_losses in losses.losses.items()
            if domain_losses is not None
        })
        if data:
            self.local_writer.add_scalars('Loss/Total', data, epoch)

    def _log_weighted_losses(self, epoch: int, losses: Losses, lambdas: Lambdas):
        # Source
        src_losses = losses.losses.get(DomainType.SOURCE)
        if src_losses is not None:
            src = _clean_scalars(self._extract_in_domain_loss_data(src_losses, lambdas.in_domain))
            if src:
                self.local_writer.add_scalars('Loss/Source', src, epoch)

        # Target
        tgt_losses = losses.losses.get(DomainType.TARGET)
        if tgt_losses is not None:
            tgt = _clean_scalars(self._extract_in_domain_loss_data(tgt_losses, lambdas.in_domain))
            if tgt:
                self.local_writer.add_scalars('Loss/Target', tgt, epoch)

    def _log_eval_losses(self, epoch: int, eval_losses: Dict[ModelTag, Loss]):
        data = {}
        for model_tag in eval_losses.keys():
            avg = eval_losses[model_tag].get_average()
            data[model_tag.short_tag] = avg
        data = _clean_scalars(data)
        if data:
            self.local_writer.add_scalars('Loss/Eval_Losses', data, epoch)

    @staticmethod
    def _extract_in_domain_loss_data(losses: DomainLosses, lambdas: InDomainLambdas):
        data = {
            'mlp_classification': (
                (losses.classification.get_average() * lambdas.classification)
                if (losses and losses.classification and losses.classification.get_average() is not None)
                else None
            ),
        }
        return {k: v for k, v in data.items() if v is not None}

    def _log_loss_type(self, epoch: int, losses: Losses):
        def _classification_avg_for(domain):
            dl = losses.losses.get(domain)  # Dict[DomainType, DomainLosses]
            if dl is None:
                return None
            cls = getattr(dl, "classification", None)
            if cls is None or not hasattr(cls, "get_average"):
                return None
            avg = cls.get_average()
            return avg if avg is not None else None

        data = {
            "mlp_classification": {
                str(DomainType.SOURCE): _classification_avg_for(DomainType.SOURCE),
                str(DomainType.TARGET): _classification_avg_for(DomainType.TARGET),
            },
        }

        filtered_data = {
            key: _clean_scalars(val)
            for key, val in data.items()
            if any(_to_float_or_none(v) is not None for v in val.values())
        }
        for loss_type, loss_values in filtered_data.items():
            if loss_values:
                self.local_writer.add_scalars(f'Loss/{loss_type}', loss_values, epoch)

    def _log_status(self, epoch: int, status: EpochTrainStatus):
        prob = _clean_scalars({'val': status.mixup_prob})
        if prob:
            self.local_writer.add_scalars('Mixup/Probability', prob, epoch)

        self.local_writer.add_scalars('Dropout/Rates', _clean_scalars(status.dropouts), epoch)
        self.local_writer.add_scalars('Layers/Unfrozen_Layers', _clean_scalars(status.num_layers), epoch)

        thresholds = {}
        if 'hard_positive_source' in status.thresholds:
            thresholds[str(DomainType.SOURCE)] = status.thresholds['hard_positive_source']
        if 'hard_positive_target' in status.thresholds:
            thresholds[str(DomainType.TARGET)] = status.thresholds['hard_positive_target']
        thresholds = _clean_scalars(thresholds)

        if thresholds:
            self.local_writer.add_scalars('Thresholds/Hard_Probability', thresholds, epoch)
            pl_data = _clean_scalars({
                'Total Pseudo Labels': sum(status.pseudo_label_counts.values()) if status.pseudo_label_counts else None,
                'Incorrect Pseudo Labels': sum(
                    status.incorrect_pseudo_label_counts.values()) if status.incorrect_pseudo_label_counts else None
            })
            if pl_data:
                self.local_writer.add_scalars('Pseudo_Labels/Counts', pl_data, epoch)

        if status.pseudo_label_thresholds is not None:
            th = _clean_scalars(status.pseudo_label_thresholds)
            if th:
                self.local_writer.add_scalars('Pseudo_Labels/Thresholds', th, epoch)

        th = _clean_scalars({'correct': status.correct_pseudo_labels, 'incorrect': status.incorrect_pseudo_labels})
        if th:
            self.local_writer.add_scalars('Pseudo_Labels/Counts', th, epoch)

        if status.pseudo_label_counts is not None:
            counts = _clean_scalars(status.pseudo_label_counts)
            if counts:
                self.local_writer.add_scalars('Pseudo_Labels/Counts_By_Category', counts, epoch)

        if status.correct_pseudo_label_counts is not None:
            correct = _clean_scalars(status.correct_pseudo_label_counts)
            if correct:
                self.local_writer.add_scalars('Pseudo_Labels/Correct_Counts_By_Category', correct, epoch)

        wrong = _clean_scalars(status.incorrect_pseudo_label_counts)
        if wrong:
            self.local_writer.add_scalars('Pseudo_Labels/Incorrect_Counts_By_Category', wrong, epoch)

    def _log_stage_status(self, epoch: int, status: EpochStageStatus):
        delta = status.delta
        neg_delta = (-delta) if _to_float_or_none(delta) is not None else None

        dvc = _clean_scalars({
            'mean_change': status.mean_change,
            'delta': delta,
            '-delta': neg_delta,
        })
        if dvc:
            self.local_writer.add_scalars('Stage_Status/Delta_vs_Change', dvc, epoch)

        var = _clean_scalars({'variance': status.variance})
        if var:
            self.local_writer.add_scalars('Stage_Status/Variance', var, epoch)

        mc = _clean_scalars({'mean_change': status.mean_change})
        if mc:
            self.local_writer.add_scalars('Stage_Status/Mean_Change', mc, epoch)

        pat = _clean_scalars({
            'patience': status.patience,
            'stabilisation_counter': status.stabilisation_counter,
        })
        if pat:
            self.local_writer.add_scalars('Stage_Status/Patience', pat, epoch)

    def _log_lambdas(self, epoch: int, lambdas: Lambdas):
        ind = _clean_scalars(vars(lambdas.in_domain))
        if ind:
            self.local_writer.add_scalars('Lambdas/In_Domain', ind, epoch)

        gen = _clean_scalars({
            str(DomainType.SOURCE): lambdas.source,
            str(DomainType.TARGET): lambdas.target
        })
        if gen:
            self.local_writer.add_scalars('Lambdas/General', gen, epoch)

    def _log_learning_rates(self, epoch: int, train_stats: EpochTrainStats):
        self.local_writer.add_scalars('Learning_Rates', _clean_scalars(train_stats.learning_rates), epoch)

    def _log_metrics(self, epoch: int, stats: Stats, label_encoder: LabelEncoder):
        # aggregate metrics
        for metric_name in ['acc', 'f1', 'precision', 'recall']:
            data = {}
            for model_tag in stats.eval_stats.keys():
                metrics = stats.eval_stats[model_tag].metrics
                key = model_tag.short_tag if model_tag.data_tag.domain != model_tag.eval_on else model_tag.data_tag.short_tag
                data[key] = getattr(metrics, metric_name)
            data = _clean_scalars(data)
            if data:
                self.local_writer.add_scalars(f'Metrics/{metric_name.capitalize()}', data, epoch)

        # per-class F1 from classification report
        for model_tag in stats.eval_stats.keys():
            cr = stats.eval_stats[model_tag].metrics.cr
            data = {}
            for label, cell in cr.items():
                if label == 'accuracy':
                    continue
                try:
                    pretty = label_encoder.inverse_transform([int(label)])[0]
                except Exception:
                    pretty = label
                data[pretty] = cell.get('f1-score', None)
            data = _clean_scalars(data)
            if data:
                selector = model_tag.short_tag if model_tag.data_tag.domain != model_tag.eval_on else model_tag.data_tag.short_tag
                self.local_writer.add_scalars(f'Metrics/CR/{selector}', data, epoch)

    def close(self):
        try:
            self.local_writer.flush()
            self.local_writer.close()
        except Exception:
            pass
