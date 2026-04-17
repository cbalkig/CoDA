from typing import Dict, Any, Optional

from configs.base.section import Section
from data.stages.stage_types import Stages


class PseudoLabelPhaseConfig:
    def __init__(self, cfg: Dict[str, Any]):
        self.op_type: Optional[str] = None
        self.initial_threshold: Optional[float] = None
        self.final_threshold: Optional[float] = None
        self.delta: Optional[float] = None

        if "type" in cfg:
            self.op_type: float = cfg["type"]

        if "initial_threshold" in cfg:
            self.initial_threshold: float = cfg["initial_threshold"]

        if "final_threshold" in cfg:
            self.final_threshold: float = cfg["final_threshold"]

        if "delta" in cfg:
            self.delta: float = cfg["delta"]


class PseudoLabelConfig:
    def __init__(self, cfg: Section):
        stage_configs: Dict[Optional[Stages], PseudoLabelPhaseConfig] = {
            None: PseudoLabelPhaseConfig(cfg.get("default"))
        }

        for stage in Stages:
            if cfg.get(stage.value) is not None:
                stage_configs[stage] = PseudoLabelPhaseConfig(cfg.get(stage.value))

        self._type: Dict[Optional[Stages], str] = {}
        self._initial_threshold: Dict[Optional[Stages], float] = {}
        self._final_threshold: Dict[Optional[Stages], float] = {}
        self._delta: Dict[Optional[Stages], float] = {}

        for key in stage_configs.keys():
            self._type[key] = stage_configs[key].op_type
            self._initial_threshold[key] = stage_configs[key].initial_threshold
            self._final_threshold[key] = stage_configs[key].final_threshold
            self._delta[key] = stage_configs[key].delta

    def type(self, stage: Optional[Stages] = None) -> str:
        return self._type[None] if (
                stage not in self._type.keys() or self._type[
            stage] is None) else \
            self._type[stage]

    def initial_threshold(self, stage: Optional[Stages] = None) -> float:
        return self._initial_threshold[None] if (
                stage not in self._initial_threshold.keys() or self._initial_threshold[
            stage] is None) else \
            self._initial_threshold[stage]

    def final_threshold(self, stage: Optional[Stages] = None) -> float:
        return self._final_threshold[None] if (
                stage not in self._final_threshold.keys() or self._final_threshold[
            stage] is None) else self._final_threshold[stage]

    def delta(self, stage: Optional[Stages] = None) -> float:
        return self._delta[None] if (
                stage not in self._delta.keys() or self._delta[stage] is None) else \
            self._delta[stage]
