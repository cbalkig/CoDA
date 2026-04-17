from typing import Optional, Dict, Any

from configs.base.section import Section
from data.stages.stage_types import Stages


class MixupCriterionPhaseConfig:
    def __init__(self, cfg: Dict[str, Any]):
        self.mixup_alpha: Optional[float] = None
        self.cutmix_alpha: Optional[float] = None
        self.mixup_prob: Optional[float] = None
        self.switch_prob: Optional[float] = None
        self.mode: Optional[str] = None
        self.label_smoothing: Optional[int] = None
        self.scheduler_mode: Optional[str] = None
        self.mixup_prob_delta: Optional[float] = None

        if "mixup_alpha" in cfg:
            self.mixup_alpha: float = cfg["mixup_alpha"]

        if "cutmix_alpha" in cfg:
            self.cutmix_alpha: float = cfg["cutmix_alpha"]

        if "mixup_prob" in cfg:
            self.mixup_prob: float = cfg["mixup_prob"]

        if "switch_prob" in cfg:
            self.switch_prob: float = cfg["switch_prob"]

        if "mode" in cfg:
            self.mode: str = cfg["mode"]

        if "label_smoothing" in cfg:
            self.label_smoothing: float = cfg["label_smoothing"]

        if "scheduler_mode" in cfg:
            self.scheduler_mode: str = cfg["scheduler_mode"]

        if "mixup_prob_delta" in cfg:
            self.mixup_prob_delta: float = cfg["mixup_prob_delta"]


class MixupCriterionConfig:
    def __init__(self, cfg: Section):
        stage_configs: Dict[Optional[Stages], MixupCriterionPhaseConfig] = {
            None: MixupCriterionPhaseConfig(cfg.get("default"))
        }

        for stage in Stages:
            if cfg.get(stage.value) is not None:
                stage_configs[stage] = MixupCriterionPhaseConfig(cfg.get(stage.value))

        self._mixup_alpha: Dict[Optional[Stages], float] = {}
        self._cutmix_alpha: Dict[Optional[Stages], float] = {}
        self._mixup_prob: Dict[Optional[Stages], float] = {}
        self._switch_prob: Dict[Optional[Stages], float] = {}
        self._mode: Dict[Optional[Stages], str] = {}
        self._label_smoothing: Dict[Optional[Stages], float] = {}
        self._scheduler_mode: Dict[Optional[Stages], str] = {}
        self._mixup_prob_delta: Dict[Optional[Stages], float] = {}

        for key in stage_configs.keys():
            self._mixup_alpha[key] = stage_configs[key].mixup_alpha
            self._cutmix_alpha[key] = stage_configs[key].cutmix_alpha
            self._mixup_prob[key] = stage_configs[key].mixup_prob
            self._switch_prob[key] = stage_configs[key].switch_prob
            self._mode[key] = stage_configs[key].mode
            self._label_smoothing[key] = stage_configs[key].label_smoothing
            self._scheduler_mode[key] = stage_configs[key].scheduler_mode
            self._mixup_prob_delta[key] = stage_configs[key].mixup_prob_delta

    def mixup_alpha(self, stage: Optional[Stages] = None) -> float:
        return self._mixup_alpha[None] if (
                    stage not in self._mixup_alpha.keys() or self._mixup_alpha[stage] is None) else \
            self._mixup_alpha[stage]

    def cutmix_alpha(self, stage: Optional[Stages] = None) -> float:
        return self._cutmix_alpha[None] if (
                    stage not in self._cutmix_alpha.keys() or self._cutmix_alpha[stage] is None) else \
            self._cutmix_alpha[stage]

    def mixup_prob(self, stage: Optional[Stages] = None) -> float:
        return self._mixup_prob[None] if (stage not in self._mixup_prob.keys() or self._mixup_prob[stage] is None) else \
            self._mixup_prob[stage]

    def switch_prob(self, stage: Optional[Stages] = None) -> float:
        return self._switch_prob[None] if (
                    stage not in self._switch_prob.keys() or self._switch_prob[stage] is None) else \
            self._switch_prob[stage]

    def mode(self, stage: Optional[Stages] = None) -> str:
        return self._mode[None] if (stage not in self._mode.keys() or self._mode[stage] is None) else self._mode[stage]

    def label_smoothing(self, stage: Optional[Stages] = None) -> float:
        return self._label_smoothing[None] if (
                    stage not in self._label_smoothing.keys() or self._label_smoothing[stage] is None) else \
            self._label_smoothing[stage]

    def scheduler_mode(self, stage: Optional[Stages] = None) -> str:
        return self._scheduler_mode[None] if (
                    stage not in self._scheduler_mode.keys() or self._scheduler_mode[stage] is None) else \
            self._scheduler_mode[stage]

    def mixup_prob_delta(self, stage: Optional[Stages] = None) -> float:
        return self._mixup_prob_delta[None] if (
                    stage not in self._mixup_prob_delta.keys() or self._mixup_prob_delta[stage] is None) else \
            self._mixup_prob_delta[stage]
