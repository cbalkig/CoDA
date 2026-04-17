import logging
import math

import bpy

from configs.line_art import LineArtConfig


class BlenderLineArtOps:
    @staticmethod
    def setup(cfg: LineArtConfig) -> None:
        if cfg.method == "freestyle":
            scn = bpy.context.scene
            view_layer = bpy.context.view_layer

            if hasattr(view_layer, "use_freestyle"):
                view_layer.use_freestyle = True
            if hasattr(scn.render, "use_freestyle"):
                scn.render.use_freestyle = True

            if hasattr(scn.render, "line_thickness_mode"):
                scn.render.line_thickness_mode = 'ABSOLUTE'
            if hasattr(scn.render, "line_thickness"):
                scn.render.line_thickness = float(cfg.thickness_px)

            fs = view_layer.freestyle_settings
            lineset = fs.linesets[0] if fs.linesets else fs.linesets.new("LineSet")

            lineset.select_by_visibility = True
            if hasattr(lineset, "visibility"):
                lineset.visibility = 'VISIBLE'
            lineset.select_by_edge_types = True
            lineset.select_silhouette = True
            lineset.select_crease = True
            lineset.select_border = False
            lineset.select_edge_mark = False
            if hasattr(lineset, "select_external_contour"):
                lineset.select_external_contour = False

            if hasattr(fs, "crease_angle"):
                fs.crease_angle = math.radians(cfg.crease_threshold_deg)

            logging.debug(f"Freestyle outlines ON (thickness={cfg.thickness_px}, crease={cfg.crease_threshold_deg}°).")
        else:
            scene = bpy.context.scene
            view_layer = bpy.context.view_layer

            # Enable Normal pass if requested
            if cfg.use_normal_pass:
                try:
                    view_layer.use_pass_normal = True
                except AttributeError:
                    scene.view_layers[view_layer.name].use_pass_normal = True

            # Transparent film so alpha edges are valid
            scene.render.film_transparent = True

            scene.use_nodes = True
            nt = scene.node_tree
            nt.nodes.clear()
            N, L = nt.nodes, nt.links

            rl = N.new("CompositorNodeRLayers")

            # Decide edge source
            if cfg.use_alpha_edges and "Alpha" in rl.outputs.keys():
                edge_source = "Alpha"
            elif cfg.use_normal_pass and "Normal" in rl.outputs.keys():
                edge_source = "Normal"
            else:
                edge_source = "Image"

            # Optional blur
            blur = None
            if cfg.blur_px > 0:
                blur = N.new("CompositorNodeBlur")
                blur.size_x = cfg.blur_px
                blur.size_y = cfg.blur_px
                blur.use_relative = False

            # Edge detection
            sobel = N.new("CompositorNodeFilter")
            sobel.filter_type = 'SOBEL'
            absv = N.new("CompositorNodeMath")
            absv.operation = 'ABSOLUTE'

            # Threshold
            thr = N.new("CompositorNodeMath")
            thr.operation = 'GREATER_THAN'
            thr.inputs[1].default_value = float(cfg.edge_threshold)

            # Morphological open: erode then dilate
            erode = None
            if cfg.open_px > 0:
                erode = N.new("CompositorNodeDilateErode")
                erode.mode = 'DISTANCE'
                erode.distance = cfg.open_px

            # Thickness
            d1 = N.new("CompositorNodeDilateErode")
            d1.mode = 'DISTANCE'
            d1.distance = max(1, cfg.thickness_px // 2)
            d2 = N.new("CompositorNodeDilateErode")
            d2.mode = 'DISTANCE'
            d2.distance = max(0, cfg.thickness_px - d1.distance)

            # Map to black lines
            ramp = N.new("CompositorNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
            ramp.color_ramp.elements[1].position = 1.0
            ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
            invert = N.new("CompositorNodeInvert")

            # Multiply lines over original
            mix = N.new("CompositorNodeMixRGB")
            mix.blend_type = 'MULTIPLY'
            mix.use_alpha = True
            mix.inputs[0].default_value = float(cfg.strength)

            # White background (to avoid transparency washout)
            white = N.new("CompositorNodeRGB")
            white.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            over = N.new("CompositorNodeAlphaOver")
            over.premul = True

            comp = N.new("CompositorNodeComposite")

            # Wire graph
            src_out = rl.outputs[edge_source]
            if blur:
                L.new(src_out, blur.inputs[0])
                L.new(blur.outputs[0], sobel.inputs[1])
            else:
                L.new(src_out, sobel.inputs[1])

            L.new(sobel.outputs[0], absv.inputs[0])
            L.new(absv.outputs[0], thr.inputs[0])

            if erode:
                L.new(thr.outputs[0], erode.inputs[0])
                L.new(erode.outputs[0], d1.inputs[0])
            else:
                L.new(thr.outputs[0], d1.inputs[0])

            L.new(d1.outputs[0], d2.inputs[0])
            L.new(d2.outputs[0], ramp.inputs[0])
            L.new(ramp.outputs[0], invert.inputs[1])

            L.new(rl.outputs["Image"], mix.inputs[1])  # base
            L.new(invert.outputs[0], mix.inputs[2])  # multiplied outlines

            # White background with alpha-over
            L.new(white.outputs[0], over.inputs[1])
            L.new(mix.outputs[0], over.inputs[2])
            L.new(over.outputs[0], comp.inputs["Image"])

            logging.debug(f"Compositor outlines ON (src={edge_source}, thickness={cfg.thickness_px}, "
                          f"threshold={cfg.edge_threshold}, blur={cfg.blur_px}, open={cfg.open_px}, strength={cfg.strength}).")
