from configparser import SectionProxy


class PostProcessConfig:
    def __init__(self, cfg: SectionProxy):
        self.min_obj_ratio = cfg.getfloat("min_obj_ratio")
