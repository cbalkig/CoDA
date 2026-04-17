from __future__ import annotations

import math
import random
from typing import List, Dict, Any

from mathutils import Euler

from configs.pose import PoseConfig
from data.xyz import XYZ
from values.rigid_transform import RigidTransformData


class BlenderPoseOps:

    # ──────────────────────────────────────────────────────────────────
    # 1.  Rigid‑body pose randomisation
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def random_rigid_transform(objs, pose_cfg: PoseConfig) -> List[Dict[str, Any]]:
        """
        Jitter location, rotation (Euler XYZ in **degrees**) and uniform scale
        of every mesh/empty in *objs*.

        Returns a “pose‑key” string so the caller can log / de‑duplicate.
        """
        sampled: list[Dict[str, Any]] = []

        for o in objs:
            if o.type not in {"MESH", "EMPTY"}:
                continue

            # ── translation ───────────────────────────────────
            loc_range = pose_cfg.location_range
            dx = random.uniform(*loc_range[0])
            dy = random.uniform(*loc_range[1])
            dz = random.uniform(*loc_range[2])
            o.location = (dx, dy, dz)

            # ── rotation ─────────────────────────────────────
            rx = random.uniform(*pose_cfg.x_rotation_range_deg)
            ry = random.uniform(*pose_cfg.y_rotation_range_deg)
            rz = random.uniform(*pose_cfg.z_rotation_range_deg)

            o.rotation_euler = Euler(
                (math.radians(rx), math.radians(ry), math.radians(rz)), "XYZ"
            )

            # ── uniform scale ────────────────────────────────
            scale_range = pose_cfg.scale_range
            s = random.uniform(*scale_range)
            o.scale = (s, s, s)

            sampled.append(
                RigidTransformData(
                    XYZ(dx, dy, dz),
                    XYZ(rx, ry, rz),
                    round(s, 3)
                ).to_dict())

        return sampled
