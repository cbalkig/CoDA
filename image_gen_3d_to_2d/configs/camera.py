from configparser import SectionProxy


class CameraConfig:
    def __init__(self, cfg: SectionProxy):
        self.azimuth_range_deg = tuple(map(int, cfg.get("azimuth_range_deg").split(",")))
        self.elevation_range_deg = tuple(map(int, cfg.get("elevation_range_deg").split(",")))
        self.radius_multiplier = tuple(map(float, cfg.get("radius_multiplier").split(",")))
        self.focal_mm_range = tuple(map(float, cfg.get("focal_mm_range").split(",")))
        self.sensor_mm = cfg.getint("sensor_mm")
        self.aspect_choices = tuple(map(float, cfg.get("aspect_choices").split(",")))
        self.fstop_range = tuple(map(float, cfg.get("fstop_range").split(",")))
        self.fit_margin = tuple(map(float, cfg.get("fit_margin").split(",")))
        self.min_area_frac = cfg.getfloat("min_area_frac")
        self.max_retries = cfg.getint("max_retries")
        self.overload_eps = cfg.getfloat("overload_eps")
        self.min_depth_multiplier = cfg.getfloat("min_depth_multiplier")
        self.max_depth_multiplier = cfg.getfloat("max_depth_multiplier")
