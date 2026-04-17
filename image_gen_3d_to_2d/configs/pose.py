from configparser import SectionProxy


class PoseConfig:
    def __init__(self, cfg: SectionProxy):
        data = tuple(map(float, cfg.get("location_range").split(",")))
        self.location_range = [(data[0], data[1]), (data[2], data[3]), (data[4], data[5])]
        self.x_rotation_range_deg = tuple(map(int, cfg.get("x_rotation_range_deg").split(",")))
        self.y_rotation_range_deg = tuple(map(int, cfg.get("y_rotation_range_deg").split(",")))
        self.z_rotation_range_deg = tuple(map(int, cfg.get("z_rotation_range_deg").split(",")))
        self.scale_range = tuple(map(float, cfg.get("scale_range").split(",")))
