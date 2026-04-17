from typing import Optional

from configs.base.section import Section
from data.stages.stage_types import Stages
from data.types.operation_type import OperationType


class GeneralConfig:
    def __init__(self, cfg: Section):
        self.tag: str = cfg.get("tag")
        self.operation: OperationType = OperationType.from_str(cfg.get("operation")) if cfg.get(
            "operation") else OperationType.TRAINING
        self.stage: Optional[Stages] = None if cfg.get("stage") is None else Stages.get_by_name(cfg.get("stage"))
        self.seed: int = cfg.getint("seed")
