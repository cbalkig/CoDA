import re
from abc import ABC, abstractmethod
from typing import AnyStr, List, Tuple

from data.file.path import StoragePath


class AbstractDatasetType(ABC):
    def __init__(self, file_pattern: AnyStr):
        self.file_pattern = re.compile(file_pattern)

    def filter(self, files: List[StoragePath]) -> List[StoragePath]:
        import logging
        logger = logging.getLogger(__name__)
        file_data: List[Tuple[StoragePath, tuple]] = []

        def _sort_key(p: StoragePath):
            return str(p.path.absolute())

        invalid_pattern_count = 0

        for file in sorted(files, key=_sort_key):
            match = self.file_pattern.match(file.path.name)
            if match is None:
                invalid_pattern_count += 1
                continue

            file_data.append((file, match.groups()))

        if invalid_pattern_count > 0:
            logger.info(f"Filtered out {invalid_pattern_count} files because they did not match the pattern '{self.file_pattern.pattern}'")

        return self._filter(file_data)

    @abstractmethod
    def _filter(self, files: List[Tuple[StoragePath, tuple]]) -> List[StoragePath]:
        pass
