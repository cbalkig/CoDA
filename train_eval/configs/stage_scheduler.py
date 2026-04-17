from configs.base.section import Section


class StageSchedulerConfig:
    def __init__(self, cfg: Section):
        self.min_patience: int = cfg.getint("min_patience")
        self.max_patience: int = cfg.getint("max_patience")
        self.warmup_epochs: int = cfg.getint("warmup_epochs")
        self.best_improve_eps: float = cfg.getfloat("best_improve_eps")
        self.post_best_grace_epochs: int = cfg.getint("post_best_grace_epochs")
