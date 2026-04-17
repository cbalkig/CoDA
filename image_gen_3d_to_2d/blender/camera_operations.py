from __future__ import annotations

import logging
import math
import random
from enum import Enum, unique
from typing import Dict, Any, List, Optional, Tuple

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from configs.camera import CameraConfig
from data.xyz import XYZ
from values.camera import CameraData


@unique
class CamAction(Enum):
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    MOVE_FORWARD = 5
    MOVE_BACKWARD = 6


class CamStatusDepth(Enum):
    BACKWARD = 0
    FORWARD = 1
    INSIDE = 2
    OK = 3


class CamStatusHorizontal(Enum):
    LEFT = 0
    HALF_LEFT = 1
    RIGHT = 2
    HALF_RIGHT = 3
    OVERSIZED = 4
    OK = 5


class CamStatusVertical(Enum):
    UP = 0
    HALF_UP = 1
    DOWN = 2
    HALF_DOWN = 3
    OVERSIZED = 4
    OK = 5


class BlenderCameraOps:
    def __init__(self, cfg: CameraConfig, corners: List[Vector], dia: float, center: Vector, resolution: int):
        self.cfg = cfg
        self.corners = corners
        self.center = center
        self.dia = dia
        self.resolution = resolution

        self.action_list: list[CamAction] = []
        self.status_list: list[tuple[CamStatusDepth, CamStatusHorizontal, CamStatusVertical]] = []

    def setup_random_camera(self) -> Tuple[Dict[str, Any], str | None]:
        scene = bpy.context.scene
        cam = BlenderCameraOps._get_or_create_camera()
        bpy.ops.view3d.camera_to_view_selected()

        # ------------ intrinsics ------------
        cam.data.lens = random.uniform(*self.cfg.focal_mm_range)
        logging.debug(f"Camera focal length: {cam.data.lens} - Range: {self.cfg.focal_mm_range}")

        cam.data.sensor_width = self.cfg.sensor_mm
        aspect = random.choice(self.cfg.aspect_choices)
        logging.debug(f"Camera aspect ratio: {aspect} - Range: {self.cfg.aspect_choices}")

        scene.render.resolution_y = self.resolution
        scene.render.resolution_x = int(self.resolution * aspect + 0.5)

        sensor_h = cam.data.sensor_width / aspect  # vertical size in mm
        fov_x = 2 * math.atan((cam.data.sensor_width / 2) / cam.data.lens)
        fov_y = 2 * math.atan((sensor_h / 2) / cam.data.lens)

        margin = random.uniform(*self.cfg.fit_margin)
        logging.debug(f"Camera margin: {margin} - Range: {self.cfg.fit_margin}")

        min_fit_dist = (self.dia / 2) / math.sin(min(fov_x, fov_y) / 2) * margin
        dyn_min_fit_dist = min_fit_dist

        az = math.radians(random.uniform(*self.cfg.azimuth_range_deg))
        el_low_deg, el_high_deg = self.cfg.elevation_range_deg
        el = math.asin(random.uniform(math.sin(math.radians(el_low_deg)), math.sin(math.radians(el_high_deg))))
        r_selected = random.uniform(*self.cfg.radius_multiplier)
        r = max(dyn_min_fit_dist, self.dia * r_selected)

        logging.debug(f"Camera azimuth: {az} - Range: {self.cfg.azimuth_range_deg}")
        logging.debug(f"Camera elevation: {el} - Range: {self.cfg.elevation_range_deg}")
        logging.debug(f"Camera radius: {r} - {r_selected} - Range: {self.cfg.radius_multiplier}")

        cam.rotation_quaternion = (self.center - cam.location).to_track_quat('-Z', 'Y')

        # ------------ retry loop ------------
        for attempt in range(self.cfg.max_retries):
            # depth-of-field + clips
            cam.data.clip_start = max(0.01 * self.dia, 0.05)
            cam.data.clip_end = (cam.location - self.center).length * 3.5

            logging.debug(f"Camera Clip Start: {cam.data.clip_start}")
            logging.debug(f"Camera Clip End: {cam.data.clip_end}")

            cam.data.dof.use_dof = True
            cam.data.dof.focus_distance = (cam.location - self.center).length
            cam.data.dof.aperture_fstop = random.uniform(*self.cfg.fstop_range)

            bpy.context.view_layer.update()

            logging.debug(f"Camera center: {self.center} - Location: {cam.location}")
            logging.debug(f"Camera rotation quaternion: {cam.rotation_quaternion}")

            ok, status, cam_action, speed = self._evaluate_view(cam, scene)

            logging.debug(
                f"[CamTry: {attempt + 1}/{self.cfg.max_retries}] - Lens: {cam.data.lens:.1f} mm - Dist: {r:.3f} - Cam Action: {cam_action} - Speed: {speed:.3f}")

            if ok:
                break

            if attempt == 0:
                if status == CamStatusDepth.INSIDE:
                    logging.debug(f"Camera is inside the object")
                    return CameraData(math.degrees(az), math.degrees(el), math.degrees(r_selected), cam.data.lens,
                                      aspect,
                                      cam.data.dof.aperture_fstop, margin,
                                      XYZ(cam.location.x, cam.location.y,
                                          cam.location.z)).to_dict(), "Incorrect cam location"
                elif status == CamStatusDepth.BACKWARD:
                    logging.debug(f"Camera is backward")
                    return CameraData(math.degrees(az), math.degrees(el), math.degrees(r_selected), cam.data.lens,
                                      aspect,
                                      cam.data.dof.aperture_fstop, margin,
                                      XYZ(cam.location.x, cam.location.y,
                                          cam.location.z)).to_dict(), "Incorrect cam location"

            # Pre‑compute orientation vectors (view, right, up) in world space.
            view_vec = (self.center - cam.location).normalized()

            right_vec = cam.matrix_world.to_quaternion() @ Vector((1, 0, 0))
            right_vec.normalize()

            up_vec = cam.matrix_world.to_quaternion() @ Vector((0, 1, 0))
            up_vec.normalize()

            logging.debug(f"Camera location (before): {cam.location}")
            logging.debug(f"Speed: {speed}")
            logging.debug(f"View vector: {view_vec}")
            logging.debug(f"Right vector: {right_vec}")
            logging.debug(f"Up vector: {up_vec}")

            if cam_action == CamAction.MOVE_BACKWARD:
                logging.debug(f"Moving backward the camera...")
                logging.debug(f"View Vector: {view_vec}")
                cam.location -= view_vec * speed
            elif cam_action == CamAction.MOVE_FORWARD:
                logging.debug(f"Moving forward the camera...")
                logging.debug(f"View Vector: {view_vec}")
                cam.location += view_vec * speed
            elif cam_action == CamAction.MOVE_RIGHT:
                logging.debug(f"Moving right the camera...")
                logging.debug(f"Right Vector: {right_vec}")
                cam.location += right_vec * speed
            elif cam_action == CamAction.MOVE_LEFT:
                logging.debug(f"Moving left the camera...")
                logging.debug(f"Right Vector: {right_vec}")
                cam.location -= right_vec * speed
            elif cam_action == CamAction.MOVE_DOWN:
                logging.debug(f"Moving down the camera...")
                logging.debug(f"Up Vector: {up_vec}")
                cam.location -= up_vec * speed
            elif cam_action == CamAction.MOVE_UP:
                logging.debug(f"Moving up the camera...")
                logging.debug(f"Up Vector: {up_vec}")
                cam.location += up_vec * speed

            logging.debug(f"Camera location (after): {cam.location}")

            # cam.rotation_quaternion = (self.center - cam.location).to_track_quat('-Z', 'Y')
            bpy.context.view_layer.update()
        else:
            logging.debug(
                f"Camera: FAILED after {self.cfg.max_retries} attempts; lens={cam.data.lens}f mm, lastDist={(cam.location - self.center).length}")
            return {}, "Max attempts reached"

        return CameraData(math.degrees(az), math.degrees(el), math.degrees(r_selected), cam.data.lens, aspect,
                          cam.data.dof.aperture_fstop, margin,
                          XYZ(cam.location.x, cam.location.y, cam.location.z)).to_dict(), None

    @staticmethod
    def _get_or_create_camera() -> bpy.types.Object:
        cam = bpy.data.objects.get("RenderCam")
        if cam is None:
            cam_data = bpy.data.cameras.new("RenderCam")
            cam = bpy.data.objects.new("RenderCam", cam_data)
            bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        return cam

    @staticmethod
    def _are_opposite(a: CamAction | None, b: CamAction | None) -> bool:
        """Return True if a and b are opposite movements."""
        opposites = {
            CamAction.MOVE_UP: CamAction.MOVE_DOWN,
            CamAction.MOVE_DOWN: CamAction.MOVE_UP,
            CamAction.MOVE_LEFT: CamAction.MOVE_RIGHT,
            CamAction.MOVE_RIGHT: CamAction.MOVE_LEFT,
            CamAction.MOVE_FORWARD: CamAction.MOVE_BACKWARD,
            CamAction.MOVE_BACKWARD: CamAction.MOVE_FORWARD,
        }
        return a is not None and b is not None and opposites.get(a) == b

    def _evaluate_view(self, cam: bpy.types.Object, scene: bpy.types.Scene) -> \
            tuple[bool, CamStatusDepth, CamAction, float]:
        xs, ys, zs = [], [], []
        for c in self.corners:
            uv_x, uv_y, uv_z = world_to_camera_view(scene, cam, c)
            xs.append(uv_x)
            ys.append(uv_y)
            zs.append(uv_z)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        logging.debug(f"Min z={min_z} - Max z={max_z}")
        logging.debug(f"Min x={min_x} - Max x={max_x}")
        logging.debug(f"Min y={min_y} - Max y={max_y}")

        move_z = None
        if min_z < 0 and max_z < 0:
            cam_status_depth = CamStatusDepth.BACKWARD
            move_z = min_z + self.cfg.min_depth_multiplier * self.dia
        elif min_z < 0 < max_z:
            cam_status_depth = CamStatusDepth.INSIDE
            move_z = min_z + self.cfg.min_depth_multiplier * self.dia
        elif min_z > 0 and max_z > 0:
            if self.cfg.min_depth_multiplier * self.dia <= min_z <= self.cfg.max_depth_multiplier * self.dia:
                cam_status_depth = CamStatusDepth.OK
            elif self.cfg.min_depth_multiplier * self.dia > min_z:
                cam_status_depth = CamStatusDepth.BACKWARD
                move_z = self.cfg.max_depth_multiplier * self.dia - min_z
            elif min_z > self.cfg.max_depth_multiplier * self.dia:
                cam_status_depth = CamStatusDepth.FORWARD
                move_z = min_z - self.cfg.min_depth_multiplier * self.dia
            else:
                raise Exception(f"Unknown Min z={min_z} - Max z={max_z}")
        else:
            raise Exception(f"Unknown Min z={min_z} - Max z={max_z}")

        move_x = None
        if min_x < -self.cfg.overload_eps and max_x < -self.cfg.overload_eps:
            cam_status_horizontal = CamStatusHorizontal.LEFT
            move_x = min_x - -self.cfg.overload_eps
        elif min_x < -self.cfg.overload_eps and max_x > 1 + self.cfg.overload_eps:
            cam_status_horizontal = CamStatusHorizontal.OVERSIZED
        elif min_x > 1 + self.cfg.overload_eps and max_x > 1 + self.cfg.overload_eps:
            cam_status_horizontal = CamStatusHorizontal.RIGHT
            move_x = max_x - (1 + self.cfg.overload_eps)
        elif min_x >= -self.cfg.overload_eps and max_x <= 1 + self.cfg.overload_eps:
            cam_status_horizontal = CamStatusHorizontal.OK
        elif min_x >= -self.cfg.overload_eps and max_x > 1 + self.cfg.overload_eps:
            cam_status_horizontal = CamStatusHorizontal.HALF_RIGHT
            move_x = max_x - (1 + self.cfg.overload_eps)
        elif min_x < -self.cfg.overload_eps and max_x <= 1 + self.cfg.overload_eps:
            cam_status_horizontal = CamStatusHorizontal.HALF_LEFT
            move_x = min_x - -self.cfg.overload_eps
        else:
            raise Exception(f"Unknown Min x={min_x} - Max x={max_x}")

        move_y = None
        if min_y < -self.cfg.overload_eps and max_y < -self.cfg.overload_eps:
            cam_status_vertical = CamStatusVertical.DOWN
            move_y = min_y - -self.cfg.overload_eps
        elif min_y < -self.cfg.overload_eps and max_y > 1 + self.cfg.overload_eps:
            cam_status_vertical = CamStatusVertical.OVERSIZED
        elif min_y > 1 + self.cfg.overload_eps and max_y > 1 + self.cfg.overload_eps:
            cam_status_vertical = CamStatusVertical.UP
            move_y = max_y - (1 + self.cfg.overload_eps)
        elif min_y >= -self.cfg.overload_eps and max_y <= 1 + self.cfg.overload_eps:
            cam_status_vertical = CamStatusVertical.OK
        elif min_y >= -self.cfg.overload_eps and max_y > 1 + self.cfg.overload_eps:
            cam_status_vertical = CamStatusVertical.HALF_UP
            move_y = max_y - (1 + self.cfg.overload_eps)
        elif min_y < -self.cfg.overload_eps and max_y <= 1 + self.cfg.overload_eps:
            cam_status_vertical = CamStatusVertical.HALF_DOWN
            move_y = min_y - -self.cfg.overload_eps
        else:
            raise Exception(f"Unknown Min y={min_y} - Max y={max_y}")

        logging.debug(
            f"Cam Status - Depth: {cam_status_depth}, Horizontal: {cam_status_horizontal}, Vertical: {cam_status_vertical}")

        action: Optional[CamAction] = None
        action_speed: float = 0.0

        loop_detected: bool = (
                len(self.action_list) >= 2
                and self._are_opposite(self.action_list[-1], self.action_list[-2])
        )

        if loop_detected:
            action = CamAction.MOVE_BACKWARD  # object behind / too close
            logging.debug(f"Loop detected: {loop_detected}.")
            action_speed = self.dia * 0.1
        elif cam_status_depth in [CamStatusDepth.BACKWARD, CamStatusDepth.INSIDE]:
            action = CamAction.MOVE_BACKWARD  # object behind / too close
            logging.debug(f"Object is behind or too close to the camera view.")
            action_speed = move_z
        else:
            if cam_status_horizontal == CamStatusHorizontal.LEFT:
                action = CamAction.MOVE_LEFT
                logging.debug(f"Object is left of the camera view.")
                action_speed = move_x
            elif cam_status_horizontal == CamStatusHorizontal.HALF_LEFT:
                action = CamAction.MOVE_LEFT
                logging.debug(f"Object is left of the camera view.")
                action_speed = move_x / 2
            elif cam_status_horizontal == CamStatusHorizontal.RIGHT:
                action = CamAction.MOVE_RIGHT
                logging.debug(f"Object is right of the camera view.")
                action_speed = move_x
            elif cam_status_horizontal == CamStatusHorizontal.HALF_RIGHT:
                action = CamAction.MOVE_RIGHT
                logging.debug(f"Object is right of the camera view.")
                action_speed = move_x / 2
            else:
                if cam_status_vertical == CamStatusVertical.DOWN:
                    action = CamAction.MOVE_DOWN
                    logging.debug(f"Object is down of the camera view.")
                    action_speed = move_y
                elif cam_status_vertical == CamStatusVertical.HALF_DOWN:
                    action = CamAction.MOVE_DOWN
                    logging.debug(f"Object is down of the camera view.")
                    action_speed = move_y / 2
                elif cam_status_vertical == CamStatusVertical.UP:
                    action = CamAction.MOVE_UP
                    logging.debug(f"Object is up of the camera view.")
                    action_speed = move_y
                elif cam_status_vertical == CamStatusVertical.HALF_UP:
                    action = CamAction.MOVE_UP
                    logging.debug(f"Object is up of the camera view.")
                    action_speed = move_y / 2
                elif cam_status_horizontal == CamStatusHorizontal.OVERSIZED or cam_status_vertical == CamStatusVertical.OVERSIZED:
                    action = CamAction.MOVE_BACKWARD
                    action_speed = self.dia * 0.1
                else:
                    if cam_status_depth == CamStatusDepth.FORWARD:
                        action = CamAction.MOVE_FORWARD
                        logging.debug(f"Object is too far from the camera view.")
                        action_speed = move_z

        # all corners inside view *and* within depth window?
        ok = (
                cam_status_horizontal == CamStatusHorizontal.OK and cam_status_vertical == CamStatusVertical.OK and cam_status_depth == CamStatusDepth.OK
        )

        self.action_list.append(action)
        self.status_list.append((cam_status_depth, cam_status_horizontal, cam_status_vertical))

        return ok, cam_status_depth, action, min(max(abs(action_speed), 0.01), 1)
