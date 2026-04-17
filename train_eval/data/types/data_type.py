from enum import Enum, auto


class DataType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
    PSEUDO_LABEL = auto()

    def __str__(self):
        return self.name.lower()
