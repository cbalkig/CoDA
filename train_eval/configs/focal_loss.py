from configs.base.section import Section


class FocalLossConfig:
    def __init__(self, cfg: Section):
        self.beta: float = cfg.getfloat("beta")
        self.gamma: float = cfg.getfloat("gamma")
        self.reduction: str = cfg.get("reduction")
        self.eps: float = cfg.getfloat("eps")
