from enum import Enum, auto


class OperationType(Enum):
    HYPERPARAMETER_TUNING = auto()
    TRAINING = auto()
    EVALUATION = auto()

    def __str__(self):
        return self.name.lower().replace('_', ' ')

    @classmethod
    def from_str(cls, value: str):
        normalized = value.strip().lower().replace(" ", "_")
        for member in cls:
            if member.name.lower() == normalized:
                return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")
