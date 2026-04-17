from pathlib import Path

from configs.base.section import Section


class EvaluationConfig:
    def __init__(self, cfg: Section):
        self.reports: bool = cfg.getboolean("reports") if cfg.get("reports") is not None else False
        self.pseudo_labels: bool = cfg.get("pseudo_labels") if cfg.get("pseudo_labels") is not None else False
        self.cleanup: bool = cfg.getboolean("cleanup") if cfg.get("cleanup") is not None else False
        self.report_folder: Path = Path(cfg.get("report_folder")) if cfg.get("report_folder") is not None else None
        self.remove_base: bool = cfg.get("remove_base") if cfg.get("remove_base") is not None else False
