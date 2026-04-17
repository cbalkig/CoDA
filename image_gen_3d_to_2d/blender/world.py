from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

import bpy
import pandas as pd
from mathutils import Vector

from utils import Utils


@contextlib.contextmanager
def mute_blender():
    old_out, old_err = os.dup(1), os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    os.close(dn)
    try:
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)


class BlenderWorld:
    def __init__(self, seed: int, number_samples: int):
        df = pd.read_csv("solid_colors.csv")
        df["%"] = df["%"] / 100.0

        sampled = df.sample(
            n=number_samples,
            weights="%",
            replace=True,
            random_state=seed
        )

        self._solid_colors: List[str] = sampled["Hex"].tolist()

        df = pd.read_csv("neutral_colors.csv")
        df["%"] = df["%"] / 100.0

        sampled = df.sample(
            n=number_samples,
            weights="%",
            replace=True,
            random_state=seed
        )

        self._neutral_colors: List[str] = sampled["Hex"].tolist()

    @property
    def scene(self) -> bpy.types.Scene:
        return bpy.context.scene

    @property
    def objects(self) -> List[bpy.types.Object]:
        return list(self.scene.objects)

    @property
    def main_objects(self) -> List[bpy.types.Object]:
        return [o for o in self.objects if o.type == "MESH"]

    def get_solid_color(self, idx: int) -> str | None:
        if idx >= len(self._solid_colors):
            return None

        return self._solid_colors[idx]

    def get_neutral_color(self, idx: int) -> str | None:
        if idx >= len(self._neutral_colors):
            return None

        return self._neutral_colors[idx]

    @staticmethod
    def reset() -> None:
        """Reset Blender to a pristine state (factory settings)."""
        bpy.ops.wm.read_factory_settings(use_empty=True)

    @staticmethod
    def cleanup(images: list[Path]) -> None:
        for img in images:
            img.unlink(missing_ok=True)

        for img in list(bpy.data.images):
            if img.users == 0 or img.name.startswith("Render Result"):
                bpy.data.images.remove(img, do_unlink=True)

    def setup_world(self, world_color: str = "World", background_color: Utils.Colour = (0.0, 0.0, 0.0, 0.0)) -> None:
        world = self.scene.world or bpy.data.worlds.new(world_color)
        bpy.context.scene.world = world
        world.use_nodes = True

        nt = world.node_tree
        nt.nodes.clear()
        o = nt.nodes.new
        out = o("ShaderNodeOutputWorld")
        cam_b = o("ShaderNodeBackground")
        blk_b = o("ShaderNodeBackground")
        lp = o("ShaderNodeLightPath")
        mix = o("ShaderNodeMixShader")

        cam_b.inputs["Color"].default_value = Utils.to_rgba(background_color)
        cam_b.inputs["Strength"].default_value = 1.0
        blk_b.inputs["Strength"].default_value = 0.0
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(blk_b.outputs["Background"], mix.inputs[1])
        nt.links.new(cam_b.outputs["Background"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    @staticmethod
    def prep_cycles(device: str) -> None:
        """Configure Cycles once at start-up – robust GPU/MPS detection."""
        prefs = bpy.context.preferences

        if device in {"GPU", "MPS"}:
            addon_prefs = prefs.addons["cycles"].preferences
            available = {d.type for d in (addon_prefs.get_devices() or []) if hasattr(d, "type")}

            for backend in ("METAL", "OPTIX", "CUDA", "HIP"):
                if backend in available:
                    addon_prefs.compute_device_type = backend
                    bpy.context.scene.cycles.device = "GPU"
                    break
            else:
                bpy.context.scene.cycles.device = "CPU"
        else:
            bpy.context.scene.cycles.device = "CPU"

        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = False
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.view_settings.look = "None"

    def clean_lights(self) -> None:
        for o in [o for o in self.objects if o.type == "LIGHT"]:
            bpy.data.objects.remove(o, do_unlink=True)

    @staticmethod
    def render(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        scene = bpy.context.scene
        scene.render.use_border = False  # UI checkbox
        try:  # clear any Ctrl-B border
            bpy.ops.render.border(clear=True)
        except Exception:
            pass

        with mute_blender():
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            logging.info(f"Rendering {path}")

    @staticmethod
    def backup_scene() -> Path:
        with mute_blender():
            tmpdir = Path(tempfile.gettempdir())
            tmpdir = tmpdir / f"{uuid.uuid4()}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(tmpdir.absolute()), copy=True)
            return tmpdir

    def restore_scene(self, path: Path, images: List[Path], remove_backup_file: bool = True) -> None:
        if not path.exists():
            raise Exception("Path does not exist")

        with mute_blender():
            bpy.ops.wm.open_mainfile(filepath=str(path.absolute()))

        if remove_backup_file:
            os.remove(path)

        self.cleanup(images)

    @staticmethod
    def refresh() -> None:
        bpy.context.view_layer.update()

    def clean_empty_objects(self, *, delete: bool = True) -> List[bpy.types.Object]:
        keep: List[bpy.types.Object] = []

        for o in self.objects:
            if o.type == "EMPTY":
                if delete:
                    logging.debug(f"Removing empty object - {o.name}")
                    bpy.data.objects.remove(o, do_unlink=True)
                else:
                    keep.append(o)
            else:
                keep.append(o)

        BlenderWorld.refresh()
        return keep

    def clean_tiny_objects(self, eps: float, *, delete: bool = True) -> List[bpy.types.Object]:
        keep: List[bpy.types.Object] = []

        for o in self.objects:
            if o.type == "MESH":
                thin, wide = min(o.dimensions), max(o.dimensions)
                if thin < eps and wide > 10 * eps:
                    logging.warning(
                        f"Removing paper-thin mesh {o.name} - Dims: {tuple(round(d, 3) for d in o.dimensions)}")
                    if delete:
                        bpy.data.objects.remove(o, do_unlink=True)
                    else:
                        keep.append(o)

                    continue

            keep.append(o)

        BlenderWorld.refresh()
        return keep

    def join_mesh_children(self, apply_modifiers=True, delete_sources=False):
        """
        Joins every direct child that is a mesh into `root_obj`.

        Parameters
        ----------
        root_obj : bpy.types.Object
            The object that will absorb the geometry.
        apply_modifiers : bool
            If True, applies all modifiers on the children before joining.
        delete_sources : bool
            If True, removes the child objects after joining to keep the scene clean.
        """
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        # Collect mesh children
        meshes = [c for c in self.objects if c.type == 'MESH']
        if not meshes:
            logging.error(f"No mesh children found.")
            raise RuntimeError("No mesh children found.")

        # Make root the active object and select the children
        root_obj = meshes[0]
        bpy.context.view_layer.objects.active = root_obj
        root_obj.select_set(True)
        for obj in meshes:
            obj.select_set(True)
            if apply_modifiers:
                for mod in obj.modifiers:
                    try:
                        bpy.context.view_layer.objects.active = obj
                        bpy.ops.object.modifier_apply(modifier=mod.name)
                    except RuntimeError:
                        logging.warning(f"Could not apply modifier {mod.name} on {obj.name}")

        # Switch active back to root and join
        bpy.context.view_layer.objects.active = root_obj
        bpy.ops.object.join()  # merges selected meshes into the active one

        # Optionally clean up
        if delete_sources:
            for obj in meshes:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception as e:
                    logging.warning(f"Could not remove object {obj.name}")

        logging.debug(f"Joined {len(meshes)} meshes into {root_obj.name}")

    def normalise_meshes(self) -> None:
        with bpy.context.temp_override(view_layer=bpy.context.view_layer):
            for obj in self.objects:
                logging.debug("Imported %s (%s)", obj.name, obj.type)
                if obj.type != "MESH":
                    continue
                if obj.parent or obj.is_instancer:
                    continue  # avoid bake errors on children/linked data

                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                obj.location = (0.0, 0.0, 0.0)

        BlenderWorld.refresh()

    def uniformise_diameter(self, target_diameter=1.0):
        meshes = [o for o in self.objects if o.type == "MESH"]
        if not meshes:
            return

        wp_corners = []
        for ob in meshes:
            wp_corners += [ob.matrix_world @ Vector(c) for c in ob.bound_box]

        min_c = Vector((min(v.x for v in wp_corners),
                        min(v.y for v in wp_corners),
                        min(v.z for v in wp_corners)))
        max_c = Vector((max(v.x for v in wp_corners),
                        max(v.y for v in wp_corners),
                        max(v.z for v in wp_corners)))

        diameter = (max_c - min_c).length
        if diameter == 0:
            return  # degenerate case - nothing to do

        scale = target_diameter / diameter
        centre = (min_c + max_c) * 0.5

        # -- shift to origin first, then scale --------------------------------
        for ob in meshes:
            ob.location -= centre
            ob.scale *= scale

        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

        BlenderWorld.refresh()
