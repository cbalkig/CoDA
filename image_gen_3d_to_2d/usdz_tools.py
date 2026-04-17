from __future__ import annotations

from pathlib import Path
from typing import List

import bpy


class USDZTools:
    """Utilities for importing USD* files into Blender and analysing their bounds."""

    @staticmethod
    def import_file(
            path: Path,
            *,
            import_cameras: bool = True,
            import_lights: bool = False,
            import_materials: bool = True,
            light_intensity_scale: float = 1.0,
            scene_unit_scale: float | None = None,
            import_scale: float = 1,
    ) -> List[bpy.types.Object]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise ValueError(f"{path} is not a USD-family file")

        if not hasattr(bpy.ops.wm, "usd_import"):
            bpy.ops.preferences.addon_enable(module="io_scene_usd")
            if not hasattr(bpy.ops.wm, "usd_import"):
                raise RuntimeError("This Blender build lacks USD import support")

        scene = bpy.context.scene
        old_scale = scene.unit_settings.scale_length
        if scene_unit_scale is not None:
            scene.unit_settings.scale_length = scene_unit_scale

        existing = set(scene.objects)

        bpy.ops.wm.usd_import(
            filepath=str(path),
            import_cameras=import_cameras,
            import_lights=import_lights,
            import_materials=import_materials,
            light_intensity_scale=light_intensity_scale,
            import_all_materials=True,
            import_meshes=True,
            import_shapes=True,  # Blender ≤4.0
            import_subdiv=True,
            scale=import_scale,
            apply_unit_conversion_scale=(scene_unit_scale is None),
        )

        # Restore global unit scale
        scene.unit_settings.scale_length = old_scale

        new_objs = [o for o in scene.objects if o not in existing]
        return new_objs
