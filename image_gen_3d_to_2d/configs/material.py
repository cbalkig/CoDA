from configparser import SectionProxy


class MaterialConfig:
    def __init__(self, cfg: SectionProxy):
        self.hue_range = tuple(map(float, cfg.get("hue_range").split(",")))
        self.sat_range = tuple(map(float, cfg.get("sat_range").split(",")))
        self.val_range = tuple(map(float, cfg.get("val_range").split(",")))
        self.specular_range = tuple(map(float, cfg.get("specular_range").split(",")))
        self.roughness_range = tuple(map(float, cfg.get("roughness_range").split(",")))
        self.metallic_range = tuple(map(float, cfg.get("metallic_range").split(",")))
        self.uv_scale_range = tuple(map(float, cfg.get("uv_scale_range").split(",")))
        self.uv_rotate_range_deg = tuple(map(int, cfg.get("uv_rotate_range_deg").split(",")))
        self.uv_offset_range = tuple(map(float, cfg.get("uv_offset_range").split(",")))
        self.uv_extension_mode = cfg.get("uv_extension_mode")
