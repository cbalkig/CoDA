from dataclasses import dataclass
from typing import List

from mathutils import Vector


@dataclass
class Bounds:
    center: Vector
    diameter: float
    corners: List[Vector]
