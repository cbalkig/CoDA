from enum import Enum


class Stages(Enum):
    TRAIN_SOURCE = "train_source"
    FINETUNE_SOURCE_DROPOUT = "finetune_source_dropout"
    FINETUNE_SOURCE_MIXUP = "finetune_source_mixup"
    EVAL_MODELS = "eval_models"
    TRAIN_TARGET = "train_target"
    TRAIN_UPPER_TARGET = "train_upper_target"

    @classmethod
    def get_by_name(cls, name: str) -> 'Stages':
        for member in cls:
            if member.value == name:
                return member

        valid = [m.value for m in cls]
        raise ValueError(f"Unknown stage name: {name!r}. Valid options: {valid}")
