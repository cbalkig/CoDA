from configparser import SectionProxy


class TextureJitterConfig:
    def __init__(self, cfg: SectionProxy):
        self.hue_delta_range = tuple(map(float, cfg.get("hue_delta_range").split(",")))
        self.sat_delta_range = tuple(map(float, cfg.get("sat_delta_range").split(",")))
        self.val_delta_range = tuple(map(float, cfg.get("val_delta_range").split(",")))
