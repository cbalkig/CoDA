from __future__ import annotations

import math
from pathlib import Path
from typing import Union, Sequence, Tuple, List, Optional

import numpy as np
from PIL import Image
from mathutils import Vector, kdtree


class Utils:
    Colour = Union[str, Sequence[float], Sequence[int]]

    @staticmethod
    def to_rgba(col: Colour) -> tuple[float, float, float, float] | tuple[float, ...]:
        """
        Convert col to a 4‑component linear RGBA tuple in 0‑1 range.
        Accepts:
        • Hex strings "#rgb", "#rgba", "#rrggbb", "#rrggbbaa"
        • 3‑ or 4‑element tuples/lists of ints (0–255) or floats (0‑1)
        """
        if isinstance(col, str):  # ── hex branch ──
            h = col.lower().lstrip("#")
            if len(h) in {3, 4}:  # "#abc" → "#aabbcc"
                h = "".join(c * 2 for c in h)
            if len(h) == 6:
                h += "ff"  # assume 100 % alpha
            if len(h) != 8:
                raise ValueError(f"Bad colour string: {col!r}")
            r, g, b, a = (int(h[i: i + 2], 16) / 255.0 for i in (0, 2, 4, 6))

            return r, g, b, a

        if not isinstance(col, Sequence) or len(col) not in {3, 4}:  # tuples / lists
            raise ValueError("Colour tuple must have 3 or 4 components")

        comps = [c / 255.0 if isinstance(c, int) and c > 1 else float(c) for c in col]
        if len(comps) == 3:
            comps.append(1.0)  # default alpha = 1

        return tuple(comps)

    @staticmethod
    def kelvin_to_rgb(k: float) -> Tuple[float, float, float]:
        """
        Convert a colour temperature in Kelvin to linear RGB (approx.).
        Valid for 1000 K – 40000 K.
        """
        k = max(1000, min(40000, k)) / 100.0
        # red
        r = 1.0 if k <= 66 else max(0, min(1, 1.292936 * (k - 60) ** -0.133204))
        # green
        if k <= 66:
            g = max(0, min(1, 0.390081 * math.log(k) - 0.631842))
        else:
            g = max(0, min(1, 1.129891 * (k - 60) ** -0.075514))
        # blue
        b = 0 if k <= 19 else max(0, min(1, 0.543206 * (k - 10) ** -0.042))
        return (r, g, b)

    @staticmethod
    def build_kdtree(points: List[Vector]) -> kdtree.KDTree:
        kd = kdtree.KDTree(len(points))
        for idx, co in enumerate(points):
            kd.insert(co, idx)
        kd.balance()
        return kd

    @staticmethod
    def is_vector_contained(lo_a: Vector, hi_a: Vector,
                            lo_b: Vector, hi_b: Vector,
                            eps: float) -> bool:
        return (
                lo_a.x >= lo_b.x - eps and hi_a.x <= hi_b.x + eps
                and lo_a.y >= lo_b.y - eps and hi_a.y <= hi_b.y + eps
                and lo_a.z >= lo_b.z - eps and hi_a.z <= hi_b.z + eps
        )

    @staticmethod
    def euclidean_gap(lo1: Vector, hi1: Vector, lo2: Vector, hi2: Vector) -> float:
        dx = max(lo2.x - hi1.x, lo1.x - hi2.x, 0.0)
        dy = max(lo2.y - hi1.y, lo1.y - hi2.y, 0.0)
        dz = max(lo2.z - hi1.z, lo1.z - hi2.z, 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def euclidean_overlap_volume(lo1: Vector, hi1: Vector, lo2: Vector, hi2: Vector) -> float:
        ox = max(0.0, min(hi1.x, hi2.x) - max(lo1.x, lo2.x))
        oy = max(0.0, min(hi1.y, hi2.y) - max(lo1.y, lo2.y))
        oz = max(0.0, min(hi1.z, hi2.z) - max(lo1.z, lo2.z))
        return ox * oy * oz

    @staticmethod
    def get_foreground_ratio(png: Path) -> float:
        img = Image.open(png).convert("RGBA")
        alpha = [px[3] for px in img.getdata()]
        bg_pixels = sum(1 for a in alpha if a > 0)  # all non-transparent pixels are foreground
        total_pixels = len(alpha)
        return bg_pixels / total_pixels if total_pixels else 0

    @staticmethod
    def is_dark_or_bright(
            img_path: Path,
            dark_value: int = 30,
            bright_value: int = 215,
            dark_pct: float = 0.60,
            bright_pct: float = 0.60,
    ) -> Tuple[Optional[str], float]:
        """
        Returns (diagnosis, median_luminance).

        * 'diagnosis' is 'too dark', 'too bright', or None.
        * Statistics are computed **only on foreground pixels**
          (alpha > 0); background transparency will not skew the result.

        Thresholds:
        - dark_value / bright_value: clip points for the tails.
        - dark_pct / bright_pct: share of pixels that must lie in a tail
          before we call the image under- or over-exposed.
        """
        im = Image.open(img_path).convert("LA")  # Luminance + Alpha
        lum, alpha = im.split()
        arr = np.asarray(lum, dtype=np.uint8)[np.asarray(alpha) > 0]

        if arr.size == 0:  # no foreground!
            return "Image has no visible pixels", 0.0

        pct_dark = (arr < dark_value).mean()
        pct_bright = (arr > bright_value).mean()
        median_val = np.median(arr)

        if pct_bright > bright_pct:
            return f"Image is too bright - {pct_bright}", float(median_val)
        if pct_dark > dark_pct:
            return f"Image is too dark - {pct_dark}", float(median_val)
        return None, float(median_val)

    @staticmethod
    def is_full_transparent(img_path: Path) -> bool:
        im = Image.open(img_path).convert("RGBA")
        alpha = np.array(im)[:, :, 3]
        return alpha.max() == 0
