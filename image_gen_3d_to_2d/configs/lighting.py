from configparser import SectionProxy
from pathlib import Path


class LightingConfig:
    def __init__(self, cfg: SectionProxy):
        self.num_lights_range = tuple(map(int, cfg.get("num_lights_range").split(",")))
        self.light_types = tuple(map(str, cfg.get("light_types").split(",")))
        self.energy_range = tuple(map(float, cfg.get("energy_range").split(",")))
        self.temperature_range = tuple(map(int, cfg.get("temperature_range").split(",")))
        self.distance_multiplier = tuple(map(float, cfg.get("distance_multiplier").split(",")))
        self.hdr_enabled = cfg.getboolean("hdr_enabled", True)
        self.hdr_dir = Path(cfg.get("hdr_dir"))
        self.hdr_rotation_range_deg = tuple(map(int, cfg.get("hdr_rotation_range_deg").split(",")))
        self.hdr_exposure_range = tuple(map(float, cfg.get("hdr_exposure_range").split(",")))
