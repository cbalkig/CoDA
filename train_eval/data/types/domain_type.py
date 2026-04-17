from enum import Enum


class DomainType(Enum):
    SOURCE = "source"
    TARGET = "target"

    def __str__(self):
        return self.name.lower()
