from data.stages.stage_base import StageBase


class Stage:
    stage: 'StageBase'
    epoch: int

    def __init__(self, stage: StageBase, epoch: int) -> None:
        self.stage = stage
        self.epoch = epoch

    def __str__(self) -> str:
        return f"Stage: {self.stage.__class__.__name__}, Epoch: {self.epoch}"
