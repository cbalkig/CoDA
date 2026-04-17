from configparser import SectionProxy


class LineArtConfig:
    def __init__(self, cfg: SectionProxy):
        self.method: str = cfg.get("method")
        self.thickness_px: int = cfg.getint("thickness")
        self.strength: float = cfg.getfloat("strength")
        self.use_normal_pass: bool = cfg.getboolean("use_normal_pass")
        self.use_alpha_edges: bool = cfg.getboolean("use_alpha_edges")
        self.edge_threshold: float = cfg.getfloat("edge_threshold")
        self.blur_px: int = cfg.getint("blur_px")
        self.open_px: int = cfg.getint("open_px")
        self.crease_threshold_deg: float = cfg.getfloat("crease_threshold_deg")
