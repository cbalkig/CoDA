from typing import List

from data.file.path import StoragePath
from data.stages.stage import Stage
from data.stages.stage_base import StageBase
from data.stages.stage_types import Stages
from util.file_util import FileUtil


class StageHistory:
    def __init__(self) -> None:
        self.history: List[Stage] = []

    def reset(self) -> None:
        self.history.clear()

    def add(self, stage: StageBase, epoch: int) -> None:
        self.history.append(Stage(stage, epoch))

    def get_current_stage(self) -> None | Stage:
        if not self.history:
            return None

        return self.history[-1]

    def get_current_stage_type(self) -> Stages:
        return self.get_current_stage().stage.stage

    def get_epoch(self, stage_type: Stages) -> int:
        for entry in self.history:
            if entry.stage.stage == stage_type:
                return entry.epoch
        raise ValueError(f"Stage {stage_type} not found in history.")

    def save(self, file_path: StoragePath):
        FileUtil().dump(self.__dict__, file_path)

    @classmethod
    def load(cls, file_path: StoragePath) -> "StageHistory":
        state_dict = FileUtil().load(file_path, weights_only=False)
        obj = cls()
        obj.__dict__.update(state_dict)
        return obj
