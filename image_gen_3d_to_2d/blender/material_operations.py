import math
import random
from colorsys import hsv_to_rgb
from typing import List, Dict, Any

import bpy

from configs.material import MaterialConfig
from configs.texture_jitter import TextureJitterConfig
from values.materials import Materials, HSV, UV


class BlenderMaterialOps:
    """All material randomisation & augmentation utilities."""

    @staticmethod
    def randomise_materials(
            objs,
            material_cfg: MaterialConfig,
            texture_jitter_cfg: TextureJitterConfig,
    ) -> List[Dict[str, Any]]:
        """Randomise every material of *objs* while respecting new config.

        * Assets **with** an Image‑Texture + UVs:
            – keep UVs intact, insert a Hue/Sat node and apply Δ‑jitter.
        * Assets **without** valid UVs/texture:
            – treat colour as procedural and randomise a UV‑Mapping transform.

        Returns a serialisable list so the caller can reproduce or
        de‑duplicate the exact settings later.
        """
        sampled: list[Dict[str, Any]] = []

        for o in objs:
            if o.type != "MESH":
                continue

            for slot in o.material_slots:
                # ── ensure Principled BSDF ────────────────────────────────
                mat = slot.material or bpy.data.materials.new(name="AutoMat")
                if mat.users > 1:  # de‑share first
                    mat = mat.copy()
                    slot.material = mat
                mat.use_nodes = True
                nt = mat.node_tree

                bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if bsdf is None:
                    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

                # ── baseline PBR values (always absolute HSV) ────────────
                h = random.uniform(*material_cfg.hue_range)
                s = random.uniform(*material_cfg.sat_range)
                v = random.uniform(*material_cfg.val_range)
                r, g, b = hsv_to_rgb(h, s, v)
                bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)

                if "Specular" in bsdf.inputs.keys():
                    bsdf.inputs["Specular"].default_value = random.uniform(
                        *material_cfg.specular_range
                    )
                    spec_val = bsdf.inputs["Specular"].default_value
                else:
                    spec_val = None  # record as missing

                # Existing knobs
                bsdf.inputs["Roughness"].default_value = random.uniform(
                    *material_cfg.roughness_range
                )
                bsdf.inputs["Metallic"].default_value = random.uniform(
                    *material_cfg.metallic_range
                )

                noise = nt.nodes.new("ShaderNodeTexNoise")
                noise.inputs["Scale"].default_value = 250  # very fine grain
                noise.inputs["Detail"].default_value = 2.0
                noise.inputs["Roughness"].default_value = 0.0

                r_min, r_max = material_cfg.roughness_range
                map_range = nt.nodes.new("ShaderNodeMapRange")
                map_range.inputs["From Min"].default_value = 0.0
                map_range.inputs["From Max"].default_value = 1.0
                map_range.inputs["To Min"].default_value = r_min
                map_range.inputs["To Max"].default_value = r_max

                nt.links.new(noise.outputs["Fac"], map_range.inputs["Value"])
                nt.links.new(map_range.outputs["Result"], bsdf.inputs["Roughness"])

                # ── texture vs procedural branch ─────────────────────────
                tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)

                # *** honour uv_extension_mode to kill UV‑seam artefacts ***
                if tex and material_cfg.uv_extension_mode:
                    tex.extension = material_cfg.uv_extension_mode.upper()

                has_uv = bool(tex and getattr(o.data, "uv_layers", None))

                if has_uv:
                    # --- (A) keep UVs, just jitter colour ----------------
                    hs_node = nt.nodes.new("ShaderNodeHueSaturation")
                    nt.links.new(hs_node.outputs[0], bsdf.inputs["Base Color"])
                    nt.links.new(tex.outputs[0], hs_node.inputs["Color"])

                    hΔ = random.uniform(*texture_jitter_cfg.hue_delta_range)
                    sΔ = random.uniform(*texture_jitter_cfg.sat_delta_range)
                    vΔ = random.uniform(*texture_jitter_cfg.val_delta_range)
                    hs_node.inputs["Hue"].default_value = 0.5 + hΔ
                    hs_node.inputs["Saturation"].default_value = sΔ
                    hs_node.inputs["Value"].default_value = vΔ

                    sampled.append(
                        Materials(
                            hsv=HSV(h + hΔ, s * sΔ, v * vΔ),
                            roughness=bsdf.inputs["Roughness"].default_value,
                            specular=spec_val,
                        ).to_dict()
                    )

                else:
                    # --- (B) procedural asset – randomise UV transform ----
                    map_node = next((n for n in nt.nodes if n.type == "MAPPING"), None)
                    if map_node is None:
                        map_node = nt.nodes.new("ShaderNodeMapping")

                    if not any(
                            isinstance(i.from_node, bpy.types.ShaderNodeTexCoord)
                            for i in nt.links
                            if i.to_node == map_node
                    ):
                        uv_node = nt.nodes.new("ShaderNodeTexCoord")
                        nt.links.new(uv_node.outputs["UV"], map_node.inputs["Vector"])

                    if tex and not any(i.from_node == map_node and i.to_node == tex for i in nt.links):
                        nt.links.new(map_node.outputs["Vector"], tex.inputs["Vector"])

                    sc = random.uniform(*material_cfg.uv_scale_range)
                    rot_deg = random.uniform(*material_cfg.uv_rotate_range_deg)
                    off = random.uniform(*material_cfg.uv_offset_range)
                    map_node.inputs["Scale"].default_value = (sc, sc, sc)
                    map_node.inputs["Rotation"].default_value[2] = math.radians(rot_deg)
                    loc = map_node.inputs["Location"].default_value
                    loc[0], loc[1] = off, off

                    sampled.append(
                        Materials(
                            hsv=HSV(h, s, v),
                            roughness=bsdf.inputs["Roughness"].default_value,
                            specular=spec_val,
                            uv=UV(sc, rot_deg, off),
                        ).to_dict()
                    )

        return sampled
