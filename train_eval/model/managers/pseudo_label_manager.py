from typing import Optional


class PseudoLabelManager:
    def __init__(self):
        self._active = None
        self._type = None
        self._initial_epoch = None
        self._current_threshold = None

        # Fixed
        self._initial_threshold: Optional[float] = None
        self._delta: Optional[float] = None
        self._final_threshold: Optional[float] = None

        self.reset()

    def reset(self):
        self._current_threshold = None
        self._initial_threshold = None
        self._final_threshold = None
        self._delta = None
        self._active = False
        self._initial_epoch = None
        self._type = None

    def _enable(self, epoch: int, op_type: str, initial_threshold: Optional[float], final_threshold: Optional[float],
                delta: Optional[float]) -> None:
        if op_type not in ['fixed', 'dynamic_top']:
            raise ValueError(f"PseudoLabelManager only support 'fixed' and 'dynamic_top', not '{op_type}'.")

        if delta is None or delta <= 0:
            raise ValueError("delta must be > 0")
        if initial_threshold is None or final_threshold is None:
            raise ValueError("initial_threshold and final_threshold are required")
        if op_type == 'fixed' and not (0.0 <= final_threshold <= initial_threshold <= 1.0):
            raise ValueError("Require 0 ≤ final_threshold ≤ initial_threshold ≤ 1")
        if op_type == 'dynamic_top' and not (0.0 <= initial_threshold <= final_threshold <= 1.0):
            raise ValueError("Require 0 ≤ initial_threshold ≤ final_threshold ≤ 1")

        self._type = op_type
        self._initial_threshold = float(initial_threshold)
        self._delta = float(delta)
        self._current_threshold = float(initial_threshold)
        self._final_threshold = float(final_threshold)

        self._active = True
        self._initial_epoch = epoch

    def _disable(self) -> None:
        self._active = False

    def add(self, epoch: int, op_type: str, initial_threshold: Optional[float],
            final_threshold: Optional[float], delta: Optional[float], threshold=None):
        if threshold is None:
            return self._enable(epoch, op_type, initial_threshold, final_threshold, delta)
        else:
            return self._disable()

    def step(self, epoch: int) -> None:
        if not self._active:
            return

        if self._initial_epoch is None:
            return

        if (epoch - self._initial_epoch) <= 1:
            return

        if self._type == 'fixed':
            self._current_threshold = max(self._current_threshold - self._delta, self._final_threshold)

        elif self._type == 'dynamic_top':
            self._current_threshold = min(self._current_threshold + self._delta, self._final_threshold)

    def completed(self) -> bool:
        if not self._active:
            return True

        if self._type == 'fixed':
            return self._current_threshold <= self._final_threshold

        elif self._type == 'dynamic_top':
            return self._current_threshold >= self._final_threshold

        return False

    @property
    def active(self) -> bool:
        return bool(self._active)

    def get(self) -> float:
        if not self._active:
            raise RuntimeError('PseudoLabelManager is disabled; call .enable(...) first')

        return self._current_threshold

    @property
    def initial_epoch(self) -> int:
        return self._initial_epoch

    @property
    def type(self) -> str:
        return self._type
