from typing import Dict, Optional

from configs.base.section import Section
from data.file.path import StoragePath


class StorageConfig:
    def __init__(self, cfg: Section):
        self.download: bool = bool(cfg.get('download')) if cfg.get('download') else False
        self.remote_folder: StoragePath = StoragePath(cfg.get("remote_folder"))
        self.local_folder: StoragePath = StoragePath(cfg.get("local_folder"))
        self.source_folder: Optional[StoragePath] = StoragePath(cfg.get("source_folder")) if cfg.get(
            "source_folder") else None
        self.target_folder: Optional[StoragePath] = StoragePath(cfg.get("target_folder")) if cfg.get(
            "target_folder") else None

        self.test_folders: Dict[str | None, StoragePath] = {}
        test_folders_config = cfg.get("test_folders")
        if test_folders_config is not None:
            for raw_key in test_folders_config.keys():
                key = raw_key.strip()
                val = test_folders_config[key].strip()
                self.test_folders[key] = StoragePath(val)

        self.destination_folder: StoragePath = StoragePath(cfg.get("destination_folder"))
        self.backup_folder: StoragePath = StoragePath(cfg.get("backup_folder"))
