from __future__ import annotations

import logging
import math
from typing import List, Optional

import bmesh
import bpy
from mathutils import Vector, kdtree

from data.bounds import Bounds
from utils import Utils


# ------------------------------------------------------------------ #
#  Blender-version-safe Vector helpers                                #
# ------------------------------------------------------------------ #
def vmin(a: Vector, b: Vector) -> Vector:
    """Component-wise minimum that works on all Blender versions."""
    return Vector((min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)))


def vmax(a: Vector, b: Vector) -> Vector:
    """Component-wise maximum that works on all Blender versions."""
    return Vector((max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)))


# ------------------------------------------------------------------ #
#  Try the C-accelerated island detector (Blender ≥ 4.0)             #
# ------------------------------------------------------------------ #
try:
    from bmesh.ops import connected_components as _bm_connected_components  # type: ignore
except Exception:  # noqa: BLE001  (older Blender – op absent)
    _bm_connected_components = None


class BlenderIslands:
    # ------------------------------------------------------------------ #
    #  Island tracing                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trace_islands(
            objs: List[bpy.types.Object],
            eps: float,
    ) -> List[dict]:
        deps = bpy.context.evaluated_depsgraph_get()

        # 1) gather islands from every mesh
        flat_islands: List[tuple[bpy.types.Object, Vector, Vector, List[Vector]]] = [
            (obj, lo, hi, verts)
            for obj in objs
            if obj.type == "MESH"
            for lo, hi, verts in BlenderIslands._count_disconnected_islands(obj, deps)
        ]
        if not flat_islands:
            return []

        # 2) lazy KD-trees (one per *other* island)
        kdtrees: list[Optional[kdtree.KDTree]] = [None] * len(flat_islands)

        def tree(j: int) -> kdtree.KDTree:
            if kdtrees[j] is None:
                kdtrees[j] = Utils.build_kdtree(flat_islands[j][3])
            return kdtrees[j]

        summaries: List[dict] = []

        # 3) pair-wise relationship classification
        for idx, (obj_i, lo_i, hi_i, verts_i) in enumerate(flat_islands):
            size_i = hi_i - lo_i
            volume_i = size_i.x * size_i.y * size_i.z
            centre_i = (lo_i + hi_i) * 0.5

            tag_child = tag_overlap = tag_touch = False
            d_min = float("inf")

            for jdx, (_, lo_j, hi_j, _) in enumerate(flat_islands):
                if jdx == idx:
                    continue

                # cheap AABB checks
                tag_child |= Utils.is_vector_contained(lo_i, hi_i, lo_j, hi_j, eps)
                gap = Utils.euclidean_gap(lo_i, hi_i, lo_j, hi_j)

                if gap <= eps:
                    if Utils.euclidean_overlap_volume(lo_i, hi_i, lo_j, hi_j) > eps:
                        tag_overlap = True
                    else:
                        tag_touch = True

                if gap >= d_min:  # cannot beat current best
                    continue

                # surface-to-surface distance
                for co in verts_i:
                    _, _, dist = tree(jdx).find(co)
                    if dist < d_min:
                        d_min = dist
                        if d_min <= eps:  # perfect touch – cannot improve
                            break
                if d_min <= eps:
                    break

            if tag_child:
                type_tag = "child"
            elif tag_overlap:
                type_tag = "overlap"
            elif tag_touch:
                type_tag = "touch"
            else:
                type_tag = "normal"

            summaries.append(
                {
                    "object": obj_i.name,
                    "island": idx + 1,
                    "type": type_tag,
                    "volume": round(float(volume_i), 3),
                    "size": tuple(round(s, 3) for s in size_i),
                    "pos": tuple(round(c, 3) for c in centre_i),
                    "dist_nearest": round(float(d_min), 6),
                }
            )

        return summaries

    # ------------------------------------------------------------------ #
    #  Per-mesh island detection                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _count_disconnected_islands(
            obj: bpy.types.Object,
            deps: Optional[bpy.types.Depsgraph] = None,
    ) -> List[tuple[Vector, Vector, List[Vector]]]:
        """
        Returns [(lo, hi, verts), …] for every vertex-connected island of `obj`
        in world coordinates.  Uses the C op when available, otherwise a
        Python flood-fill.
        """
        if obj.type != "MESH":
            return []

        eval_obj = obj.evaluated_get(deps) if deps else obj
        mesh = eval_obj.to_mesh()

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()

        M = obj.matrix_world
        islands: List[tuple[Vector, Vector, List[Vector]]] = []

        # ---- Fast path: Blender ≥ 4.0 ----------------------------------
        if _bm_connected_components is not None:
            sets = _bm_connected_components(bm, verts=bm.verts)["components"]
            for comp in sets:
                co_ws = [M @ v.co for v in comp]
                xs, ys, zs = zip(*co_ws)
                lo = Vector((min(xs), min(ys), min(zs)))
                hi = Vector((max(xs), max(ys), max(zs)))
                islands.append((lo, hi, co_ws))

        # ---- Fallback: manual flood-fill -------------------------------
        else:
            visited: set[int] = set()
            for v in bm.verts:
                if v.index in visited:
                    continue

                stack = [v]
                lo = Vector((float("inf"),) * 3)
                hi = Vector((float("-inf"),) * 3)
                verts: List[Vector] = []

                while stack:
                    cur = stack.pop()
                    if cur.index in visited:
                        continue
                    visited.add(cur.index)

                    co_w = M @ cur.co
                    verts.append(co_w)
                    lo = vmin(lo, co_w)
                    hi = vmax(hi, co_w)

                    for e in cur.link_edges:
                        nxt = e.other_vert(cur)
                        if nxt.index not in visited:
                            stack.append(nxt)

                islands.append((lo, hi, verts))

        bm.free()
        eval_obj.to_mesh_clear()
        return islands

    # ------------------------------------------------------------------ #
    #  High-level bounding-box computation                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def evaluate_islands_and_bounds(
            objs: List[bpy.types.Object],
            eps: float,
            z_thresh: float = 2.5,
            min_core: int = 1,
            remove_outliers: bool = False,
    ) -> Bounds:
        island_summaries = BlenderIslands._trace_islands(objs, eps)
        if not island_summaries:
            logging.warning("No island summaries found")
            return Bounds(Vector((0, 0, 0)), 0.0, [])

        core = [r for r in island_summaries if r["type"] != "child"]
        candidates = core if len(core) >= min_core else island_summaries

        centres = [Vector(r["pos"]) for r in candidates]
        centroid = sum(centres, Vector()) / len(centres)

        dists = [(c - centroid).length for c in centres]
        mu = sum(dists) / len(dists)
        sigma = math.sqrt(sum((d - mu) ** 2 for d in dists) / len(dists)) or 1e-9

        def is_outlier(rec) -> bool:
            d = (Vector(rec["pos"]) - centroid).length
            return abs((d - mu) / sigma) > z_thresh

        filtered = [
            r
            for r in island_summaries
            if not is_outlier(r) or not remove_outliers
        ]
        if not filtered:
            logging.warning("Everything is an outlier")
            return Bounds(Vector((0, 0, 0)), 0.0, [])

        mins = Vector((float("inf"),) * 3)
        maxs = Vector((float("-inf"),) * 3)
        for rec in filtered:
            pos = Vector(rec["pos"])
            size = Vector(rec["size"])
            mins = vmin(mins, pos - size * 0.5)
            maxs = vmax(maxs, pos + size * 0.5)

        centre = (mins + maxs) * 0.5
        diameter = (maxs - mins).length
        corners = [
            Vector((mins.x, mins.y, mins.z)),
            Vector((mins.x, mins.y, maxs.z)),
            Vector((mins.x, maxs.y, mins.z)),
            Vector((mins.x, maxs.y, maxs.z)),
            Vector((maxs.x, mins.y, mins.z)),
            Vector((maxs.x, mins.y, maxs.z)),
            Vector((maxs.x, maxs.y, mins.z)),
            Vector((maxs.x, maxs.y, maxs.z)),
        ]
        return Bounds(centre, diameter, corners)
