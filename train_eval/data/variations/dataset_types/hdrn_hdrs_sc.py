from typing import Dict, List, Tuple

from data.file.path import StoragePath
from data.variations.dataset_types.abstract_dataset_types import AbstractDatasetType


class DatasetType(AbstractDatasetType):
    def __init__(self):
        super().__init__(file_pattern=r'^(?:(solid_color|hdr_nature|hdr_studio)_)?([0-9a-f]{32})_(\d+)\.png$')
        self.counts: Dict[str, int] = {
            'hdr_nature': 120,
            'hdr_studio': 60,
            'solid_color': 60,
            'sculpture': 0,
        }

    def _filter(self, files: List[Tuple[StoragePath, tuple]]) -> List[StoragePath]:
        import logging
        logger = logging.getLogger(__name__)
        filtered_files = []
        counts: Dict[str, Dict[str, int]] = {}
        filtered_out_counts: Dict[str, int] = {}

        for file, groups in files:
            family: str = groups[0]
            model_id: str = groups[1]

            if family not in counts:
                counts[family] = {}

            if model_id not in counts[family]:
                counts[family][model_id] = 0

            if counts[family][model_id] >= self.counts[family]:
                filtered_out_counts[family] = filtered_out_counts.get(family, 0) + 1
                continue

            filtered_files.append(file)
            counts[family][model_id] += 1

        total_filtered_out = sum(filtered_out_counts.values())
        if total_filtered_out > 0:
            logger.info(f"Filtered out {total_filtered_out} files because they exceeded the per-model limit:")
            for family, count in filtered_out_counts.items():
                logger.info(f"  - {family}: {count} files (limit was {self.counts[family]})")

        return filtered_files
