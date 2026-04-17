import importlib
import os
from typing import Dict, Optional

from configs.base.section import Section
from data.variations.dataset_types.abstract_dataset_types import AbstractDatasetType


class DatasetConfig:
    def __init__(self, cfg: Section):
        val = cfg.get("source_cross_val_k")
        if val is None:
            self.source_cross_val_k = None
        else:
            self.source_cross_val_k = [int(x.strip()) for x in str(val).split(",") if x.strip()]

        val = cfg.get("target_cross_val_k")
        if val is None:
            self.target_cross_val_k = None
        else:
            self.target_cross_val_k = [int(x.strip()) for x in str(val).split(",") if x.strip()]

        self.dataset_type: Optional[AbstractDatasetType] = None
        if cfg.get('dataset_type'):
            module = importlib.import_module(f"data.variations.dataset_types.{cfg.get('dataset_type')}")
            self.dataset_type: AbstractDatasetType = getattr(module, "DatasetType")()

        self.data_augmentations: Dict[str, str] = {}
        data_augmentations = cfg.get('data_augmentations')
        if data_augmentations is not None:
            for raw_key in data_augmentations.keys():
                key = str(raw_key).strip()
                val = data_augmentations[raw_key].strip()
                self.data_augmentations[key] = val

        self.num_workers: int = cfg.getint("num_workers") if cfg.get("num_workers") is not None else os.cpu_count()
