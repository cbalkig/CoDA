from __future__ import annotations

import configparser
import logging
import sys
from pathlib import Path

from configs.blender import BlenderConfig
from configs.camera import CameraConfig
from configs.default import DefaultConfig
from configs.lighting import LightingConfig
from configs.line_art import LineArtConfig
from configs.material import MaterialConfig
from configs.pose import PoseConfig
from configs.post_process import PostProcessConfig
from configs.render import RenderConfig
from configs.texture_jitter import TextureJitterConfig


class ConfigLoader:
    """Read *config.cfg* and expose all values as attributes."""

    # ------------------------------------------------------------------
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            logging.critical("No cfg at %s", path)
            sys.exit(1)

        cp = configparser.ConfigParser()
        cp.read(path)

        self.path: Path = path

        self.default_cfg = DefaultConfig(cp["default"])
        self.blender_cfg = BlenderConfig(cp["blender"])
        self.render_cfg = RenderConfig(cp["render"])
        self.post_process_cfg = PostProcessConfig(cp["post_process"])
        self.lighting_cfg = LightingConfig(cp["lighting"])
        self.camera_cfg = CameraConfig(cp["camera"])
        self.pose_cfg = PoseConfig(cp["pose"])
        self.material_cfg = MaterialConfig(cp["material"])
        self.texture_jitter_cfg = TextureJitterConfig(cp["texture_jitter"])
        self.line_art_cfg = LineArtConfig(cp["line_art"])
