# ----------------------------------------------------------------------
# renderer.py – now with **six** independent augmentation passes
# ----------------------------------------------------------------------
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import bpy

from blender.camera_operations import BlenderCameraOps
from blender.domain_randomization_operations import BlenderDomainRandomizationOps
from blender.islands import BlenderIslands
from blender.lighting_operations import BlenderLightingOps
from blender.line_art_operations import BlenderLineArtOps
from blender.material_operations import BlenderMaterialOps
from blender.pose_operations import BlenderPoseOps
from blender.sculpture_art_operations import BlenderSculptureArtOps
from blender.domain_randomization_operations import BlenderDomainRandomizationOps
from blender.gray_operations import BlenderGrayOps
from blender.world import mute_blender, BlenderWorld
from config_loader import ConfigLoader
from data.backgrounds import Backgrounds, BackgroundType
from render_logger import RenderLogger
from usdz_tools import USDZTools
from utils import Utils


class USDZRenderer:
    # ------------------------------------------------------------------
    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg

        self.num_solid_color = int(cfg.render_cfg.samples * cfg.render_cfg.solid_color_ratio)
        self.num_hdr = cfg.render_cfg.samples - self.num_solid_color

        logging.info(f'Number of samples:\n\tSolid colors: {self.num_solid_color}\tHDR: {self.num_hdr}')
        logging.info(f'HDR path: {cfg.lighting_cfg.hdr_dir}')

        self.world = BlenderWorld(self.cfg.default_cfg.seed, self.num_solid_color)

    @contextmanager
    def _temporary_film_transparent(self, enabled: bool = True):
        prev = bpy.context.scene.render.film_transparent
        bpy.context.scene.render.film_transparent = enabled
        try:
            yield
        finally:
            bpy.context.scene.render.film_transparent = prev

    @contextmanager
    def _temporary_flat_world(self, rgba=(0.0, 0.0, 0.0, 1.0), strength: float = 0.0):
        w = bpy.context.scene.world
        w.use_nodes = True
        nt = w.node_tree
        nt.nodes.clear()
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.new("ShaderNodeOutputWorld")
        bg.inputs["Color"].default_value = rgba
        bg.inputs["Strength"].default_value = strength  # 0 disables world lighting
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

        try:
            yield
        finally:
            self.world.setup_world(self.cfg.blender_cfg.world_color, self.cfg.blender_cfg.white_background_color)

    def get_object_ratio(self, image_path: Path) -> float:
        backup = self.world.backup_scene()
        with self._temporary_film_transparent(True):
            self.world.render(image_path)

        if Utils.is_full_transparent(image_path):
            self.world.render(image_path)

        self.world.restore_scene(backup, [])
        return Utils.get_foreground_ratio(image_path)

    def is_dark_or_bright(self, image_path: Path) -> tuple[str | None, float]:
        backup = self.world.backup_scene()
        with self._temporary_flat_world((0.0, 0.0, 0.0, 1.0), strength=0.0):
            BlenderSculptureArtOps.setup(self.world.objects)
            self.world.render(image_path)

        self.world.restore_scene(backup, [])
        return Utils.is_dark_or_bright(image_path, dark_value=5, dark_pct=0.9, bright_pct=0.3)

    def render_model(self, model: Path, dest_dir: Path, category: str, md5: str, log: RenderLogger) -> int:
        logging.warning(f"Rendering model {model}")

        backgrounds = Backgrounds(dest_dir, category)

        self.world.reset()
        self.world.prep_cycles(self.cfg.render_cfg.device.upper())
        self.world.setup_world(self.cfg.blender_cfg.world_color, self.cfg.blender_cfg.white_background_color)

        if self.cfg.default_cfg.debug:
            USDZTools.import_file(model, import_lights=False, import_cameras=False)
        else:
            with mute_blender():
                USDZTools.import_file(model, import_lights=False, import_cameras=False)

        self.world.clean_empty_objects()

        logging.debug(f"Number of meshes: {len(self.world.objects)}")
        if len(self.world.objects) > 1:
            self.world.join_mesh_children()

        self.world.normalise_meshes()
        self.world.uniformise_diameter()

        self.world.clean_tiny_objects(eps=self.cfg.blender_cfg.tiny_obj_eps)

        num_completed_samples = 0

        def _render() -> bool:
            nonlocal num_completed_samples

            if self.cfg.render_cfg.sculpture_art or self.cfg.render_cfg.domain_randomization:
                solid_color = self.world.get_neutral_color(num_completed_samples)
            else:
                solid_color = self.world.get_solid_color(num_completed_samples)

            image_path: Path = backgrounds.get_path(BackgroundType.HDR, md5, num_completed_samples)
            if BackgroundType.SOLID_COLOR in backgrounds.keys and solid_color is not None:
                image_path = backgrounds.get_path(BackgroundType.SOLID_COLOR, md5, num_completed_samples)

            if log.has(image_path):
                num_completed_samples += 1
                return True

            pose_k = BlenderPoseOps.random_rigid_transform(self.world.objects, self.cfg.pose_cfg)

            if self.cfg.render_cfg.sculpture_art:
                BlenderSculptureArtOps.setup(self.world.objects)
                mat_k = None
            elif self.cfg.render_cfg.domain_randomization:
                BlenderDomainRandomizationOps.setup(self.world.objects)
                mat_k = None
            elif self.cfg.render_cfg.gray:
                BlenderGrayOps.setup(self.world.objects)
                mat_k = None
            else:
                mat_k = BlenderMaterialOps.randomise_materials(self.world.objects, self.cfg.material_cfg,
                                                               self.cfg.texture_jitter_cfg)

            bounds = BlenderIslands.evaluate_islands_and_bounds(self.world.objects, eps=self.cfg.blender_cfg.island_eps)
            logging.debug(
                f"Bounds - Center: {bounds.center}, Diameter: {bounds.diameter:.2f}, Corners: {bounds.corners}")

            self.world.clean_lights()
            lgt_k = BlenderLightingOps.setup_random_lighting(bounds, self.cfg.lighting_cfg, self.cfg.render_cfg,
                                                             solid_color)

            backup = self.world.backup_scene()

            ratio: Optional[float] = None
            dark_bright_value: Optional[float] = None
            for i in range(10):
                cam_k, error = BlenderCameraOps(self.cfg.camera_cfg, bounds.corners, bounds.diameter, bounds.center,
                                                self.cfg.render_cfg.resolution).setup_random_camera()

                if error is not None:
                    log.record(md5, str(model), str(image_path), bounds, 0, -1, cam_k, lgt_k,
                               pose_k, mat_k, error)

                    logging.debug(f"Failed to generate camera for model {model}")
                    self.world.restore_scene(backup,
                                             backgrounds.get_paths(backgrounds.keys, md5, num_completed_samples),
                                             remove_backup_file=False)
                    continue

                object_ratio = self.get_object_ratio(
                    backgrounds.get_path(BackgroundType.TEMPORARY, md5, num_completed_samples))

                if object_ratio < self.cfg.post_process_cfg.min_obj_ratio:
                    log.record(md5, str(model), str(image_path), bounds, round(object_ratio, 2),
                               (-1 if dark_bright_value is None else round(dark_bright_value, 2)), cam_k, lgt_k, pose_k,
                               mat_k, "Too small")

                    logging.debug(
                        f"Ignoring generated image of the model - (ratio {object_ratio:.2f}) - Cam Settings: {cam_k['camera']} - Light Settings: {lgt_k['light']}")
                    self.world.restore_scene(backup,
                                             backgrounds.get_paths(backgrounds.keys, md5, num_completed_samples),
                                             remove_backup_file=False)
                    continue

                if self.cfg.render_cfg.gray:
                    # Gray emission is always fixed at 0.5 — brightness check is irrelevant.
                    dark_or_bright, dark_bright_value = None, -1.0
                else:
                    dark_or_bright, dark_bright_value = self.is_dark_or_bright(
                        backgrounds.get_path(BackgroundType.TEMPORARY, md5, num_completed_samples))
                if dark_or_bright is not None:
                    log.record(md5, str(model), str(image_path), bounds, round(object_ratio, 2),
                               round(dark_bright_value, 2), cam_k, lgt_k, pose_k, mat_k, dark_or_bright)

                    logging.debug(
                        f"Ignoring generated image of the model - ({dark_or_bright}) - Cam Settings: {cam_k['camera']} - Light Settings: {None if lgt_k is None else lgt_k['light']}")
                    self.world.restore_scene(backup,
                                             backgrounds.get_paths(backgrounds.keys, md5, num_completed_samples))
                    return False

                if self.cfg.render_cfg.line_art:
                    BlenderLineArtOps.setup(self.cfg.line_art_cfg)

                if self.cfg.render_cfg.sculpture_art:
                    BlenderSculptureArtOps.setup(self.world.objects)

                self.world.render(image_path)

                log.record(md5, str(model), str(image_path), bounds, round(object_ratio, 2),
                           round(dark_bright_value, 2),
                           cam_k,
                           lgt_k, pose_k, mat_k)

                extra_bgs = set(self.cfg.render_cfg.backgrounds) - {BackgroundType.HDR, BackgroundType.SOLID_COLOR}
                for bt in extra_bgs:
                    extra_path = backgrounds.get_path(bt, md5, num_completed_samples)  # same index as main
                    if bt == BackgroundType.TRANSPARENT:
                        with self._temporary_film_transparent(True):
                            self.world.render(extra_path)
                    elif bt in (BackgroundType.WHITE, BackgroundType.BLACK):
                        rgba = (1.0, 1.0, 1.0, 1.0) if bt == BackgroundType.WHITE else (0.0, 0.0, 0.0, 1.0)
                        with self._temporary_flat_world(rgba=rgba, strength=1.0):
                            self.world.render(extra_path)

                self.world.restore_scene(backup,
                                         backgrounds.get_paths(backgrounds.keys - self.cfg.render_cfg.backgrounds, md5,
                                                               num_completed_samples))

                num_completed_samples += 1
                return True

            log.record(md5, str(model), str(image_path), bounds, round(object_ratio, 2) if ratio is not None else -1,
                       round(dark_bright_value, 2) if dark_bright_value is not None else -1, image_path, lgt_k, pose_k,
                       mat_k, "Will resample")

            logging.debug(
                f"Ignoring generated image of the model - (will resample) - Cam Settings: {cam_k['camera']} - Light Settings: {None if lgt_k is None else lgt_k['light']}")
            self.world.restore_scene(backup, backgrounds.get_paths(backgrounds.keys, md5, num_completed_samples))
            return False

        while num_completed_samples < self.cfg.render_cfg.samples:
            _render()

        self.world.reset()
        return num_completed_samples
