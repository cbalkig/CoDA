from configparser import SectionProxy


class BlenderConfig:
    def __init__(self, cfg: SectionProxy):
        self.world_color = cfg.get("world_color")
        self.white_background_color = tuple(map(float, cfg.get("white_background_color").split(",")))

        self.tiny_obj_eps = cfg.getfloat("tiny_obj_eps")
        self.island_eps = cfg.getfloat("island_eps")
