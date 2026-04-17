from __future__ import annotations

import math
import random
from itertools import chain
from typing import Dict, Any, Tuple, Optional

import bpy
from mathutils import Vector

from configs.lighting import LightingConfig
from configs.render import RenderConfig
from data.bounds import Bounds
from data.xyz import XYZ
from utils import Utils
from values.light import LightKeys, LightData


class BlenderLightingOps:
    """Static helpers for procedural lighting + colour‑management."""

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------
    @staticmethod
    def setup_random_lighting(
            bounds: Bounds,
            lighting_cfg: LightingConfig,
            render_cfg: RenderConfig,
            background_color: Optional[str] = None,
    ) -> Dict[str, Any]:
        scene = bpy.context.scene

        # ------------------------------------------------------------------
        # 1) Clear previous lights & world
        # ------------------------------------------------------------------
        for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
            bpy.data.objects.remove(o, do_unlink=True)

        world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()

        # World φ(nodes): Texture  →  Background  →  Output
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

        # ------------------------------------------------------------------
        # 2) Environment/HDRI
        # ------------------------------------------------------------------
        hdr_name = "None"
        rot_deg = 0.0

        hdr_files = list(chain(
            lighting_cfg.hdr_dir.glob("*.hdr"),
            lighting_cfg.hdr_dir.glob("*.exr"),
        ))
        if hdr_files:
            tex = nt.nodes.new("ShaderNodeTexEnvironment")
            hdr = random.choice(hdr_files)
            hdr_name = hdr.name
            tex.image = bpy.data.images.load(str(hdr), check_existing=True)
            nt.links.new(tex.outputs["Color"], bg.inputs["Color"])

            # Random Y‑axis rotation (Z Euler in radians)
            mapping = nt.nodes.new("ShaderNodeMapping")
            tc = nt.nodes.new("ShaderNodeTexCoord")
            nt.links.new(tc.outputs["Generated"], mapping.inputs["Vector"])
            nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            rot_deg = random.uniform(*lighting_cfg.hdr_rotation_range_deg)
            mapping.inputs["Rotation"].default_value[2] = math.radians(rot_deg)

        env_exposure = random.uniform(*lighting_cfg.hdr_exposure_range)
        bg.inputs["Strength"].default_value = env_exposure

        # ------------------------------------------------------------------
        # 3) Direct lamps
        # ------------------------------------------------------------------
        n_lamps = random.randint(*lighting_cfg.num_lights_range)
        light_keys = []

        # Scale lamp energy based on object size.
        base_diam = getattr(lighting_cfg, "reference_diameter_m", 0.10)  # 10 cm default
        scale = max(bounds.diameter / base_diam, 0.01)  # avoid div‑by‑zero & tiny values

        for i in range(n_lamps):
            ltype = random.choice(lighting_cfg.light_types)
            data = bpy.data.lights.new(f"L{i}", type=ltype)

            # Base energy then scaled
            base_energy = random.uniform(*lighting_cfg.energy_range)
            energy = base_energy * scale
            energy_clip = getattr(lighting_cfg, "energy_clip_max", None)
            if energy_clip is not None:
                energy = min(energy, energy_clip)
            data.energy = energy

            # Kelvin → sRGB
            kelvin = random.uniform(*lighting_cfg.temperature_range)
            data.color = Utils.kelvin_to_rgb(kelvin)

            light = bpy.data.objects.new(f"L{i}", data)
            bpy.context.collection.objects.link(light)

            # Position lamps on a sphere around the object
            phi = random.uniform(0.0, 2 * math.pi)
            theta = random.uniform(0.0, math.pi)
            r = bounds.diameter * random.uniform(*lighting_cfg.distance_multiplier)
            light.location = bounds.center + Vector(
                (
                    r * math.sin(theta) * math.cos(phi),
                    r * math.sin(theta) * math.sin(phi),
                    r * math.cos(theta),
                )
            )

            # Point lamp at the centre
            light.rotation_euler = (bounds.center - light.location).to_track_quat("-Z", "Y").to_euler()

            distances: Dict[str, float] = {}
            for obj in bpy.data.objects:
                if obj.type == "MESH":
                    distances[obj.name] = (light.location - obj.location).length

            light_keys.append(
                LightKeys(light_type=ltype, energy=energy, temperature=kelvin, distance_to_objects=distances,
                          location=XYZ(light.location.x, light.location.y, light.location.z)))

        # ------------------------------------------------------------------
        # 4) Colour‑management (Filmic + exposure) & clamping
        # ------------------------------------------------------------------
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = getattr(render_cfg, "exposure", 0.0)

        scene.cycles.sample_clamp_direct = getattr(lighting_cfg, "clamp_direct", 0.0)
        scene.cycles.sample_clamp_indirect = getattr(lighting_cfg, "clamp_indirect", 0.0)

        if background_color is not None:
            BlenderLightingOps._override_camera_background(Utils.to_rgba(background_color))

        # ------------------------------------------------------------------
        # 5) Return a compact representation for logs
        # ------------------------------------------------------------------
        return LightData(hdr_name, rot_deg, env_exposure, light_keys).to_dict()

    @staticmethod
    def _override_camera_background(rgba: Tuple[float, float, float, float]) -> None:
        """Make HDRI invisible **only** to the camera while preserving it for lighting.

        Implementation details:
        * `Is Camera Ray` → 1.0 for primary eye rays, 0.0 for all indirect rays.
        * We feed **HDR** into *Shader-1* (index 0) and **Const BG** into *Shader-2* (index 1).
          Because **Mix Shader** computes `out = (1-fac)·S1 + fac·S2`, setting *Fac*=1 on
          camera rays yields *Const BG*. All other rays (Fac=0) still see the HDRI.
        """
        world = bpy.context.scene.world
        nt = world.node_tree
        scene = bpy.context.scene

        # ------------------------------------------------------------------
        # Reuse nodes if they already exist to avoid unbounded growth.
        # ------------------------------------------------------------------
        lp = next((n for n in nt.nodes if n.type == "LIGHT_PATH"), None) or nt.nodes.new("ShaderNodeLightPath")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")

        # Assume the *first* Background that is *linked into the graph* is the HDRI.
        hdr_bg = next((n for n in nt.nodes if n.type == "BACKGROUND" and n.outputs[0].is_linked), None)
        if hdr_bg is None:
            raise RuntimeError("World shader graph does not contain a linked HDR Background node")

        # Constant background – reuse if we added one earlier.
        const_bg = next((n for n in nt.nodes if n.name.startswith("ConstBG")), None) or nt.nodes.new(
            "ShaderNodeBackground")
        const_bg.name = "ConstBG"
        const_bg.inputs["Color"].default_value = rgba
        const_bg.inputs["Strength"].default_value = 1.0

        # Mix shader – reuse if present, else create.
        mix = next((n for n in nt.nodes if n.name == "CameraBGMix"), None) or nt.nodes.new("ShaderNodeMixShader")
        mix.name = "CameraBGMix"

        # ------------------------------------------------------------------
        # Wire up: Fac -> IsCameraRay, Shader1 -> HDR, Shader2 -> ConstBG
        # ------------------------------------------------------------------
        # Clear existing links to mix to avoid duplicates.
        for link in list(mix.inputs[1].links) + list(mix.inputs[2].links):
            nt.links.remove(link)
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(hdr_bg.outputs["Background"], mix.inputs[1])  # non-camera rays
        nt.links.new(const_bg.outputs["Background"], mix.inputs[2])  # camera rays

        # Ensure mix feeds the World Output.
        for link in list(out.inputs["Surface"].links):
            nt.links.remove(link)
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

        # Optional: tidy node positions so graph stays readable.
        const_bg.location = hdr_bg.location + Vector((0, -200))

        mix.location = hdr_bg.location + Vector((200, 0))

        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
