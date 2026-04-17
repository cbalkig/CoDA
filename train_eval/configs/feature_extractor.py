from typing import Optional, Dict, Any

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LRScheduler

from configs.base.section import Section
from data.model_spec import ModelSpec
from data.stages.stage_types import Stages


class FeatureExtractorPhaseConfig:
    def __init__(self, cfg: Dict[str, Any]):
        self.drop_rate: Optional[float] = None
        self.initial_drop_rate: Optional[float] = None
        self.peak_drop_rate: Optional[float] = None
        self.drop_path_rate: Optional[float] = None
        self.initial_drop_path_rate: Optional[float] = None
        self.peak_drop_path_rate: Optional[float] = None
        self.dropout_delta: Optional[float] = None
        self.optimizer_learning_rate: Optional[float] = None

        if "drop_rate" in cfg:
            self.drop_rate: float = cfg["drop_rate"]

        if "initial_drop_rate" in cfg:
            self.initial_drop_rate: float = cfg["initial_drop_rate"]

        if "peak_drop_rate" in cfg:
            self.peak_drop_rate: float = cfg["peak_drop_rate"]

        if "drop_path_rate" in cfg:
            self.drop_path_rate: float = cfg["drop_path_rate"]

        if "initial_drop_path_rate" in cfg:
            self.initial_drop_path_rate: float = cfg["initial_drop_path_rate"]

        if "peak_drop_path_rate" in cfg:
            self.peak_drop_path_rate: float = cfg["peak_drop_path_rate"]

        if "dropout_delta" in cfg:
            self.dropout_delta: float = cfg["dropout_delta"]

        if "optimizer_learning_rate" in cfg:
            self.optimizer_learning_rate: float = float(cfg["optimizer_learning_rate"])


class FeatureExtractorConfig:
    def __init__(self, cfg: Section):
        self.optimizer_type: type[Optimizer] = AdamW if cfg.get("optimizer") == "AdamW" else None
        if self.optimizer_type is None:
            raise ValueError("No optimizer configured")

        self.scheduler_type: type[LRScheduler] = ReduceLROnPlateau if cfg.get(
            "scheduler") == "ReduceLROnPlateau" else CosineAnnealingLR if cfg.get(
            "scheduler") == "CosineAnnealingLR" else None
        if self.scheduler_type is None:
            raise ValueError("No scheduler configured")

        self.scheduler_min_learning_rate: float = cfg.getfloat("scheduler_min_learning_rate")
        self.scheduler_cycle_epochs: Optional[int] = cfg.getint("scheduler_cycle_epochs")
        self.scheduler_mode: Optional[str] = cfg.get("scheduler_mode")
        self.scheduler_factor: Optional[float] = cfg.get("scheduler_factor")
        self.scheduler_patience: Optional[int] = cfg.get("scheduler_patience")
        self.scheduler_cooldown: Optional[int] = cfg.get("scheduler_cooldown")
        self.optimizer_weight_decay: float = cfg.getfloat("optimizer_weight_decay")

        self.unfrozen_all_layers: bool = cfg.getboolean("unfrozen_all_layers")
        self.timm_model: ModelSpec = ModelSpec(cfg.get("timm_model"))
        self.pretrained: bool = cfg.getboolean("pretrained")

        self.layer_wise_lr_decay_gamma: Optional[float] = cfg.getfloat("layer_wise_lr_decay_gamma")

        stage_configs: Dict[Optional[Stages], FeatureExtractorPhaseConfig] = {
            None: FeatureExtractorPhaseConfig(cfg.get("default"))
        }

        for stage in Stages:
            if cfg.get(stage.value) is not None:
                stage_configs[stage] = FeatureExtractorPhaseConfig(cfg.get(stage.value))

        self._drop_rate: Dict[Optional[Stages], float] = {}
        self._initial_drop_rate: Dict[Optional[Stages], float] = {}
        self._peak_drop_rate: Dict[Optional[Stages], float] = {}
        self._drop_path_rate: Dict[Optional[Stages], float] = {}
        self._initial_drop_path_rate: Dict[Optional[Stages], float] = {}
        self._peak_drop_path_rate: Dict[Optional[Stages], float] = {}
        self._dropout_delta: Dict[Optional[Stages], float] = {}
        self._optimizer_learning_rate: Dict[Optional[Stages], float] = {}

        for key in stage_configs.keys():
            self._drop_rate[key] = stage_configs[key].drop_rate
            self._initial_drop_rate[key] = stage_configs[key].initial_drop_rate
            self._peak_drop_rate[key] = stage_configs[key].peak_drop_rate
            self._drop_path_rate[key] = stage_configs[key].drop_path_rate
            self._initial_drop_path_rate[key] = stage_configs[key].initial_drop_path_rate
            self._peak_drop_path_rate[key] = stage_configs[key].peak_drop_path_rate
            self._dropout_delta[key] = stage_configs[key].dropout_delta
            self._optimizer_learning_rate[key] = stage_configs[key].optimizer_learning_rate

    def drop_rate(self, stage: Optional[Stages] = None) -> float:
        return self._drop_rate[None] if (stage not in self._drop_rate.keys() or self._drop_rate[stage] is None) else \
            self._drop_rate[stage]

    def initial_drop_rate(self, stage: Optional[Stages] = None) -> float:
        return self._initial_drop_rate[None] if (
                stage not in self._initial_drop_rate.keys() or self._initial_drop_rate[stage] is None) else \
            self._initial_drop_rate[
                stage]

    def peak_drop_rate(self, stage: Optional[Stages] = None) -> float:
        return self._peak_drop_rate[None] if (
                stage not in self._peak_drop_rate.keys() or self._peak_drop_rate[stage] is None) else \
            self._peak_drop_rate[
                stage]

    def drop_path_rate(self, stage: Optional[Stages] = None) -> float:
        return self._drop_path_rate[None] if (
                stage not in self._drop_path_rate.keys() or self._drop_path_rate[stage] is None) else \
            self._drop_path_rate[
                stage]

    def initial_drop_path_rate(self, stage: Optional[Stages] = None) -> float:
        return self._initial_drop_path_rate[None] if (
                stage not in self._initial_drop_path_rate.keys() or self._initial_drop_path_rate[stage] is None) else \
            self._initial_drop_path_rate[
                stage]

    def peak_drop_path_rate(self, stage: Optional[Stages] = None) -> float:
        return self._peak_drop_path_rate[None] if (
                stage not in self._peak_drop_path_rate.keys() or self._peak_drop_path_rate[stage] is None) else \
            self._peak_drop_path_rate[
                stage]

    def dropout_delta(self, stage: Optional[Stages] = None) -> float:
        return self._dropout_delta[None] if (
                stage not in self._dropout_delta.keys() or self._dropout_delta[stage] is None) else self._dropout_delta[
            stage]

    def optimizer_learning_rate(self, stage: Optional[Stages] = None) -> float:
        return self._optimizer_learning_rate[None] if (
                stage not in self._optimizer_learning_rate.keys() or self._optimizer_learning_rate[stage] is None) else \
            self._optimizer_learning_rate[
                stage]
