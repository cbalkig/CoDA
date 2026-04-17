from abc import abstractmethod, ABC
from typing import Optional


class EpochStageBase(ABC):
    @property
    @abstractmethod
    def next_substage(self) -> 'EpochStageBase':
        pass

    @abstractmethod
    def run(self) -> Optional[bool]:
        pass
