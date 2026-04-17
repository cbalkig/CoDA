from typing import Dict, Any

from configs.base.section import Section


class TrainingConfig:
    def __init__(self, cfg: Section):
        self.force_cpu: bool = cfg.getboolean("force_cpu")
        self.gpu_id: int = cfg.getint("gpu_id")
        self.train_batch_size: int = cfg.getint("train_batch_size")
        self.eval_batch_size: int = cfg.getint("eval_batch_size")
        self.lambdas: Dict[str, Any] = cfg.get("lambdas")
