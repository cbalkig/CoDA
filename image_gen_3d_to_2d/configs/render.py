from configparser import SectionProxy
from typing import List

from data.backgrounds import BackgroundType


class RenderConfig:
    def __init__(self, cfg: SectionProxy):
        self.solid_color_ratio: float = cfg.getfloat("solid_color_ratio")
        bg_str = cfg.get("backgrounds", fallback="")
        self.backgrounds: List[BackgroundType] = []
        if bg_str:
            backgrounds = tuple(map(str, bg_str.split(",")))
            for b in backgrounds:
                self.backgrounds.append(BackgroundType.from_value(b.strip()))

        self.resolution = cfg.getint("resolution")
        self.device = cfg.get("device")
        self.samples = cfg.getint("samples")

        self.line_art = cfg.getboolean("line_art", False)
        self.sculpture_art = cfg.getboolean("sculpture_art", False)
        self.domain_randomization = cfg.getboolean("domain_randomization", False)
        self.gray = cfg.getboolean("gray", False)
