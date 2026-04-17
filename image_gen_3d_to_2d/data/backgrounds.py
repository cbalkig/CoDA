from enum import Enum
from pathlib import Path
from typing import Dict, List


class BackgroundType(Enum):
    HDR = 'hdr'
    SOLID_COLOR = 'solid_color'
    TRANSPARENT = 'transparent'
    WHITE = 'white'
    BLACK = 'black'
    TEMPORARY = 'temporary'

    @classmethod
    def from_value(cls, b: str) -> "BackgroundType":
        if b is None:
            raise ValueError("BackgroundType cannot be None")
        try:
            return cls(b.strip().lower())
        except ValueError:
            raise ValueError(f"Invalid BackgroundType value: {b}")


class Backgrounds:
    def __init__(self, parent_dir: Path, category: str):
        self._backgrounds: Dict[BackgroundType, Path] = {}

        for background in BackgroundType:
            self._backgrounds[background] = parent_dir / background.value / category

    @property
    def keys(self):
        return self._backgrounds.keys()

    def get_path(self, background: BackgroundType, md5: str, produced: int) -> Path:
        return self._backgrounds[background] / f"{md5}_{produced:06d}.png"

    def get_paths(self, background_types: List[BackgroundType], md5: str, produced: int) -> List[Path]:
        paths = []
        for background_type in background_types:
            paths.append(self.get_path(background_type, md5, produced))
        return paths
