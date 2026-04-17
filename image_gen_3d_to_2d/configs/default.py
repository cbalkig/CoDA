from configparser import SectionProxy
from pathlib import Path


class DefaultConfig:
    def __init__(self, cfg: SectionProxy):
        self.cleanup = cfg.getboolean("cleanup", True)
        self.debug = cfg.getboolean("debug", True)
        self.seed: int = cfg.getint("seed")
        self.source_folder = Path(cfg.get("source_folder"))
        self.log_file = Path(cfg.get("log_file"))
        self.destination_folder = Path(cfg.get("destination_folder"))
        self.success_csv_path = Path(cfg.get("success_csv_path"))
        self.failures_csv_path = Path(cfg.get("failures_csv_path"))
