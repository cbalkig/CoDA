from enum import Enum
from typing import Dict

from data.types.domain_type import DomainType
from data.types.model_type import ModelType


class DropoutType(Enum):
    DROPOUT = "dropout"
    DROPOUT_PATH = "dropout_path"


class DropoutManager:
    def __init__(self):
        self._active = None
        self._initial_epoch = None
        self._current_drop_outs = None
        self._initial_drop_outs = None
        self._deltas = None
        self._peak_drop_outs = None

        self.reset()

    def reset(self):
        self._current_drop_outs = {}
        self._peak_drop_outs = {}
        self._deltas = {}
        self._active = False
        self._initial_epoch = None

    @staticmethod
    def _ensure_nested(d: Dict, *keys):
        for k in keys:
            if k not in d:
                d[k] = {}
            d = d[k]
        return d

    def _enable(self, epoch, domain_type, model_type, dropout_type,
                initial_drop_out: float, peak_drop_out: float, delta: float) -> None:
        self._active = True

        self._initial_epoch = epoch
        self._ensure_nested(self._peak_drop_outs, domain_type, model_type)
        self._ensure_nested(self._deltas, domain_type, model_type)
        self._ensure_nested(self._current_drop_outs, domain_type, model_type)
        self._peak_drop_outs[domain_type][model_type][dropout_type] = float(peak_drop_out)
        self._deltas[domain_type][model_type][dropout_type] = float(delta)
        self._current_drop_outs[domain_type][model_type][dropout_type] = float(initial_drop_out)

    def _disable(self) -> None:
        self._active = False

    def add(self, epoch, domain_type, model_type, dropout_type,
            drop_out=None, initial_drop_out=None, peak_drop_out=None, delta=None):
        if drop_out is None:
            return self._enable(epoch, domain_type, model_type, dropout_type,
                                initial_drop_out, peak_drop_out, delta)
        else:
            return self._disable()

    def step(self, epoch: int) -> None:
        if not self._active:
            return

        if self._initial_epoch is None:
            return

        if (epoch - self._initial_epoch) <= 1:
            return

        for domain_type, models in self._current_drop_outs.items():
            for model_type, dropouts in models.items():
                for dropout_type, drop_out in dropouts.items():
                    self._current_drop_outs[domain_type][model_type][dropout_type] = min(
                        drop_out + self._deltas[domain_type][model_type][dropout_type],
                        self._peak_drop_outs[domain_type][model_type][dropout_type])

    def completed(self) -> bool:
        if not self._active:
            return True

        completed = True
        for domain_type, models in self._current_drop_outs.items():
            for model_type, dropouts in models.items():
                for dropout_type, drop_out in dropouts.items():
                    completed = completed and drop_out >= self._peak_drop_outs[domain_type][model_type][dropout_type]

        return completed

    @property
    def active(self) -> bool:
        return bool(self._active)

    def get(self, domain_type: DomainType, model_type: ModelType, dropout_type: DropoutType) -> float:
        if not self._active:
            raise RuntimeError('DropoutManager is disabled; call .enable(...) first')

        return self._current_drop_outs[domain_type][model_type][dropout_type]
