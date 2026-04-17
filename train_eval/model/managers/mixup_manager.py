from typing import Optional


class MixupManager:
    def __init__(self):
        self._active = None
        self._initial_epoch = None
        self._current_mixup_prob = None
        self._mode = None
        self._peak_seen = None
        self._peak_mixup_prob = None
        self._delta = None

        self.reset()

    def reset(self):
        self._active = False
        self._current_mixup_prob = 0

    def add(self, epoch: int, mixup_prob: float, scheduler_mode: Optional[str] = None,
            mixup_prob_delta: Optional[float] = None) -> None:
        self._current_mixup_prob = mixup_prob

        if self._current_mixup_prob == 0.0:
            self._active = False
            return

        self._active = True
        self._initial_epoch = epoch
        self._mode: str = scheduler_mode
        self._peak_seen: bool = False

        if self._mode == "decreasing":
            self._current_mixup_prob: float = mixup_prob
        elif self._mode == "increasing":
            self._current_mixup_prob: float = 0
            self._peak_mixup_prob: float = mixup_prob
        elif self._mode == "cyclic":
            self._current_mixup_prob: float = 0
            self._peak_mixup_prob: float = mixup_prob
        elif self._mode == "fixed":
            self._current_mixup_prob: float = mixup_prob
        else:
            raise NotImplementedError

        self._delta: float = mixup_prob_delta

    def step(self, epoch: int) -> None:
        if (epoch - self._initial_epoch) <= 1:
            return

        if self._mode == "decreasing":
            self._current_mixup_prob = max(self._current_mixup_prob - self._delta, 0)
        elif self._mode == "increasing":
            self._current_mixup_prob = min(self._current_mixup_prob + self._delta, self._peak_mixup_prob)
        elif self._mode == "cyclic":
            if not self._peak_seen:
                self._current_mixup_prob = min(self._current_mixup_prob + self._delta, self._peak_mixup_prob)
            else:
                self._current_mixup_prob = max(self._current_mixup_prob - self._delta, 0)

            self._peak_seen = self._peak_seen or self._current_mixup_prob >= self._peak_mixup_prob
        elif self._mode == "fixed":
            return
        else:
            raise NotImplementedError

    def completed(self) -> bool:
        if self._mode == "decreasing":
            return self._current_mixup_prob == 0
        elif self._mode == "increasing":
            return self._current_mixup_prob == self._peak_mixup_prob
        elif self._mode == "cyclic":
            return self._peak_seen and self._current_mixup_prob == 0
        elif self._mode == "fixed":
            return True
        else:
            raise NotImplementedError

    @property
    def mixup_prob(self) -> float:
        return self._current_mixup_prob
