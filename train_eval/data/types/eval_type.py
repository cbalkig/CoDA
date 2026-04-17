from enum import Enum, auto


class EvalType(Enum):
    LOCAL = auto()
    SHARED = auto()

    def __str__(self):
        return self.name.lower()
