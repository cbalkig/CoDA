import logging
from typing import Optional, List, Dict

from configs.base.configs import Configs
from data.data_tag import DataTag
from data.file.path import StoragePath
from data.model_tag import ModelTag
from data.stages.stage_status import StageStatus
from data.stats.epoch_stage_status import EpochStageStatus
from util.file_util import FileUtil


class NetworkStabilityScheduler:
    """
    WARMUP → IMPROVING → {PLATEAU | FLUCTUATING | DIVERGING} → STOP

    Per-tag series:
      • losses[tag] : lower is better
      • f1s[tag]    : higher is better

    All statistics (warm-up calibration, sliding windows, signed % changes,
    EWMA mean_change, variance, delta, state decisions and STOP) operate on the
    concatenated pool across all provided tags and both metric types, with signs
    unified so that negative change means improvement.
    """

    # ─────────────────────────────────────────────── initialisation ──────────
    def __init__(self) -> None:
        # Configs & validation
        min_patience = Configs().stage_scheduler.min_patience
        max_patience = Configs().stage_scheduler.max_patience
        warmup_epochs = Configs().stage_scheduler.warmup_epochs

        if min_patience < 1:
            raise ValueError("min_patience must be ≥ 1")
        if max_patience < min_patience:
            raise ValueError("max_patience must be ≥ min_patience")
        if warmup_epochs < 3:
            raise ValueError("warmup_epochs must be ≥ 3 to gather statistics")

        self.min_patience = min_patience
        self.max_patience = max_patience
        self.warmup_epochs = warmup_epochs

        # Optional weighting knobs
        self.metric_type_weights: Dict[str, float] = {"loss": 0.5, "f1": 0.5}
        self.tag_weights: Dict[DataTag, float] = {}  # e.g., emphasize specific validation tags

        # Early-stop (F1-only) in addition to stabilization-based STOP
        self.best_improve_eps: float = Configs().stage_scheduler.best_improve_eps
        self.post_best_grace_epochs: int = Configs().stage_scheduler.post_best_grace_epochs

        # Runtime state
        self.reset()
        self._dynamic_params_initialised: bool = False
        self._warmup_loss_hist: Dict[DataTag, List[float]] = {}
        self._warmup_f1_hist: Dict[DataTag, List[float]] = {}
        self.last_stage_change_epoch: int = 0

        # Best trackers
        self._best_loss: Dict[DataTag, float] = {}
        self._best_f1: Dict[DataTag, float] = {}
        self._epochs_since_best_f1: int = 0
        self._last_best_epoch: Optional[int] = None

        # Baseline variance captured in warm-up (for FLUCTUATING detection)
        self.var_baseline: Optional[float] = None

    # ─────────────────────────────────────────────── public API ──────────────
    def step(
            self,
            current_epoch: int,
            losses: Dict[ModelTag, float],
            f1s: Dict[ModelTag, float],
    ) -> None:
        """
        losses: dict mapping DataTag -> loss value  (lower is better)
        f1s:    dict mapping DataTag -> f1 value    (higher is better)
        """
        if not isinstance(losses, dict) or not isinstance(f1s, dict):
            raise ValueError("losses and f1s must be dicts keyed by DataTag")

        # Helpers for consistent logging
        def _fmt(v: Optional[float], p: int = 6) -> str:
            return f"{v:.{p}f}" if v is not None else "N/A"

        def _tag_name(tag) -> str:
            return getattr(tag, "short_tag", getattr(tag, "tag", str(tag)))

        def _log() -> None:
            sl = ", ".join(f"{_tag_name(t)}:{_fmt(float(v), 4)}" for t, v in list(losses.items())[:4])
            sf = ", ".join(f"{_tag_name(t)}:{_fmt(float(v), 4)}" for t, v in list(f1s.items())[:4])
            logging.info(
                f"Epoch:{current_epoch} - status:{self.stage_status.name} "
                f"- mean_change:{_fmt(self.mean_change)} - delta:{_fmt(self.delta)} "
                f"- variance:{_fmt(self.variance)} - stab_ctr:{self.stabilisation_counter} "
                f"- f1_esb:{self._epochs_since_best_f1}/{self.max_patience} "
                f"- losses[{len(losses)}]: {sl} - f1s[{len(f1s)}]: {sf}"
            )

        # 1) Warm-up: gather both metrics per tag
        if not self._dynamic_params_initialised:
            for tag, v in losses.items():
                self._warmup_loss_hist.setdefault(tag, []).append(float(v))
            for tag, v in f1s.items():
                self._warmup_f1_hist.setdefault(tag, []).append(float(v))

            ref_len = 0
            if self._warmup_loss_hist:
                ref_len = max(ref_len, max(len(seq) for seq in self._warmup_loss_hist.values()))
            if self._warmup_f1_hist:
                ref_len = max(ref_len, max(len(seq) for seq in self._warmup_f1_hist.values()))

            if ref_len < self.warmup_epochs:
                self.stage_status = StageStatus.WARMUP
                _log()
                return

            self._auto_calibrate()  # derives base_delta, loss_window, alpha, var_baseline, etc.
            self.stage_status = StageStatus.IMPROVING
            _log()
            return  # analysis starts next epoch

        # 2) Maintain sliding windows (loss & f1, per tag)
        for tag, v in losses.items():
            self.prev_loss_series.setdefault(tag, []).append(float(v))
            if len(self.prev_loss_series[tag]) > self.loss_window:
                self.prev_loss_series[tag].pop(0)
        for tag, v in f1s.items():
            self.prev_f1_series.setdefault(tag, []).append(float(v))
            if len(self.prev_f1_series[tag]) > self.loss_window:
                self.prev_f1_series[tag].pop(0)

        have_any_series = any(
            len(s) >= 2
            for s in list(self.prev_loss_series.values()) + list(self.prev_f1_series.values())
        )
        if not have_any_series:
            self.stage_status = StageStatus.IMPROVING
            _log()
            return

        # 2b) Update bests & F1-only patience
        improved_f1 = False

        # losses: lower is better (does NOT affect F1 patience)
        for tag, v in losses.items():
            v = float(v)
            prev_best = self._best_loss.get(tag)
            if prev_best is None or (v < prev_best - self.best_improve_eps):
                self._best_loss[tag] = v
                # optional: you could also reset stabilisation_counter for loss bests if desired

        # f1: higher is better (controls the patience)
        for tag, v in f1s.items():
            v = float(v)
            prev_best = self._best_f1.get(tag)
            if prev_best is None or (v > prev_best + self.best_improve_eps):
                self._best_f1[tag] = v
                improved_f1 = True
                self._last_best_epoch = current_epoch

        self._epochs_since_best_f1 = 0 if improved_f1 else (self._epochs_since_best_f1 + 1)

        # Fresh F1 improvement cancels stabilization build-up
        if improved_f1:
            self.stabilisation_counter = 0

        # 3) Signed percentage changes (concat all tags × metrics)
        combined_changes: List[float] = []

        # losses: lower is better → negative Δ% means improvement
        for tag, arr in self.prev_loss_series.items():
            if len(arr) >= 2:
                raw = self._pct_changes(arr)
                signed = [x for x in raw]
                w = self.metric_type_weights.get("loss", 1.0) * self.tag_weights.get(tag, 1.0)
                combined_changes.extend([c * w for c in signed])

        # f1: higher is better → flip sign so increase → negative (improvement)
        for tag, arr in self.prev_f1_series.items():
            if len(arr) >= 2:
                raw = self._pct_changes(arr)
                signed = [-x for x in raw]
                w = self.metric_type_weights.get("f1", 1.0) * self.tag_weights.get(tag, 1.0)
                combined_changes.extend([c * w for c in signed])

        if not combined_changes:
            self.stage_status = StageStatus.IMPROVING
            _log()
            return

        mean_pc = sum(combined_changes) / len(combined_changes)
        var_pc = sum((c - mean_pc) ** 2 for c in combined_changes) / len(combined_changes)

        self.variance = var_pc

        # EWMA of mean percentage change (combined)
        if self.mean_change is None:
            self.mean_change = mean_pc
        else:
            self.mean_change = self.alpha * mean_pc + (1 - self.alpha) * self.mean_change

        # Volatility-scaled delta
        self.delta = self.base_delta * (1 + var_pc)

        # 4) State decision (on combined stats)
        base_var = self.var_baseline if self.var_baseline is not None else self.variance
        var_thresh = (base_var * 1.25) or 1e-9  # guard against zero

        if self.mean_change <= -self.delta and var_pc < var_thresh:
            self.stage_status = StageStatus.IMPROVING
        elif abs(self.mean_change) < self.delta:
            self.stage_status = StageStatus.PLATEAU if var_pc < var_thresh else StageStatus.FLUCTUATING
        else:
            self.stage_status = StageStatus.DIVERGING

        # 5) STOP criterion
        # (A) Stabilisation-based STOP (also requires ≥ no_improve_patience F1-quiet epochs)
        if self.stage_status in (StageStatus.PLATEAU, StageStatus.FLUCTUATING, StageStatus.DIVERGING):
            if abs(self.mean_change) <= self.delta * self.confidence_margin:
                self.stabilisation_counter += 1
            else:
                self.stabilisation_counter = 0

        _log()

    def check_termination(self, current_epoch: int) -> bool:
        should_stop_stabilised = (
                self.stage_status in (StageStatus.PLATEAU, StageStatus.FLUCTUATING, StageStatus.DIVERGING,
                                      StageStatus.STOP)
                and abs(self.mean_change) <= self.delta * self.confidence_margin
                and self.stabilisation_counter >= self.stabilisation_epochs
                and (current_epoch - self.last_stage_change_epoch) >= self.cooldown
                and self._epochs_since_best_f1 >= max(self.max_patience, self.post_best_grace_epochs)
        )

        should_stop_patience = (
                self._epochs_since_best_f1 >= self.max_patience
                and (current_epoch - self.last_stage_change_epoch) >= self.cooldown
        )

        if should_stop_stabilised or should_stop_patience:
            return True

        return False

    def change_stage(self, current_epoch: int) -> None:
        self.last_stage_change_epoch = current_epoch
        self.reset(keep_dynamic_params=True)

    # ─────────────────────────────────────────────── utilities ───────────────
    def reset(self, *, keep_dynamic_params: bool = False) -> None:
        """Clear sliding-window stats; optionally keep volatility parameters."""
        self.prev_loss_series: Dict[DataTag, List[float]] = {}
        self.prev_f1_series: Dict[DataTag, List[float]] = {}
        self.mean_change: Optional[float] = None
        self.stabilisation_counter: int = 0
        self.variance: Optional[float] = None
        self.patience: int = self.min_patience

        if not keep_dynamic_params:
            self.base_delta: float = 0.0
            self.delta: float = 0.0
            self.loss_window: int = 1
            self.alpha: float = 0.5
            self.cooldown: int = 1
            self.stabilisation_epochs: int = 1
            self.confidence_margin: float = 1.0
            self._dynamic_params_initialised = False
            self._warmup_loss_hist = {}
            self._warmup_f1_hist = {}
            self._best_loss = {}
            self._best_f1 = {}
            self._epochs_since_best_f1 = 0
            self._last_best_epoch = None
            self.var_baseline = None

        self.stage_status: StageStatus = StageStatus.WARMUP

    def get_stage_status(self) -> EpochStageStatus:
        return EpochStageStatus(
            mean_change=self.mean_change,
            delta=self.delta,
            variance=self.variance,
            patience=self.patience,
            stabilisation_counter=self.stabilisation_counter,
        )

    def save(self, path: StoragePath) -> None:
        # NOTE: If FileUtil uses JSON, dicts keyed by DataTag may need custom (de)serialization.
        # If it pickles, you're fine. Adjust if persistence is required across processes.
        FileUtil().dump(self.__dict__, path)

    @classmethod
    def load(cls, path: StoragePath) -> "NetworkStabilityScheduler":
        obj = cls.__new__(cls)
        obj.__dict__.update(FileUtil().load(path, weights_only=False))
        return obj

    # ─────────────────────────────── internal helpers ────────────────────────
    @staticmethod
    def _pct_changes(arr: List[float]) -> List[float]:
        eps = 1e-8
        # If previous value (p) is effectively zero, simply return the difference
        # or a clamped value to avoid massive percentage spikes.
        changes = []
        for p, c in zip(arr[:-1], arr[1:]):
            if abs(p) < eps:
                # Avoid division by zero. If c is also small, 0 change.
                # If c is large, just cap it or treat as raw delta.
                changes.append(0.0 if abs(c) < eps else 1.0)
            else:
                changes.append((c - p) / (abs(p) + eps))
        return changes

    def _auto_calibrate(self) -> None:
        """Derive adaptive hyper-parameters from warm-up stats across all tags & metrics."""
        signed: List[float] = []

        # loss: keep sign (negative = improvement)
        for tag, seq in self._warmup_loss_hist.items():
            if len(seq) >= 2:
                w = self.metric_type_weights.get("loss", 1.0) * self.tag_weights.get(tag, 1.0)
                signed.extend([c * w for c in self._pct_changes(seq)])

        # f1: flip sign (increase → negative = improvement)
        for tag, seq in self._warmup_f1_hist.items():
            if len(seq) >= 2:
                w = self.metric_type_weights.get("f1", 1.0) * self.tag_weights.get(tag, 1.0)
                signed.extend([-c * w for c in self._pct_changes(seq)])

        if not signed:
            signed = [0.0]

        # Typical absolute change
        mean_abs = sum(abs(x) for x in signed) / len(signed)

        # Baseline variance for fluctuation detection
        mu = sum(signed) / len(signed)
        var_signed = sum((s - mu) ** 2 for s in signed) / len(signed)
        self.var_baseline = max(var_signed, 1e-12)

        # Base delta proportional to typical absolute change
        self.base_delta = max(0.001, mean_abs * 0.5)
        self.delta = self.base_delta

        # Window inversely proportional to volatility, clamped to [3, 10]
        self.loss_window = max(3, min(10, int(1 / (mean_abs + 1e-6))))

        # EWMA smoothing factor: 2 / (N + 1)
        self.alpha = 2 / (self.loss_window + 1)

        # Heuristics for cooldown & stabilisation (generalized from original)
        self.cooldown = max(1, self.min_patience)
        self.stabilisation_epochs = max(3, int(self.min_patience / 2))
        self.confidence_margin = min(2.0, 1.0 + var_signed * 10)

        # Initialize sliding windows from warm-up tails
        self._dynamic_params_initialised = True
        self.prev_loss_series = {tag: seq[-self.loss_window:] for tag, seq in self._warmup_loss_hist.items()}
        self.prev_f1_series = {tag: seq[-self.loss_window:] for tag, seq in self._warmup_f1_hist.items()}
        self._warmup_loss_hist.clear()
        self._warmup_f1_hist.clear()

        logging.debug(
            "StageScheduler - auto-calibrated - base_delta: %.3e - window: %d - alpha: %.3f - "
            "cooldown: %d - stab_epochs: %d - conf_margin: %.2f - var_baseline: %.3e",
            self.base_delta,
            self.loss_window,
            self.alpha,
            self.cooldown,
            self.stabilisation_epochs,
            self.confidence_margin,
            self.var_baseline,
        )
