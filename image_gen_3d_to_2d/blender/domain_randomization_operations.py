import bpy
import random


class BlenderDomainRandomizationOps:
    @staticmethod
    def setup(objs):
        for o in objs:
            if o.type != "MESH":
                continue

            # Create a brand new material for domain randomization
            mat = bpy.data.materials.new(name="DomainRandomizationMat")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()

            out = nt.nodes.new("ShaderNodeOutputMaterial")
            diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
            
            # Link diffuse to output
            nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
            
            # Choose a random texture type
            tex_type = random.choice(["Noise", "Checker", "Voronoi", "Magic"])
            
            if tex_type == "Noise":
                tex = nt.nodes.new("ShaderNodeTexNoise")
                tex.inputs["Scale"].default_value = random.uniform(1.0, 50.0)
                tex.inputs["Detail"].default_value = random.uniform(0.0, 15.0)
                tex.inputs["Distortion"].default_value = random.uniform(0.0, 5.0)
            elif tex_type == "Checker":
                tex = nt.nodes.new("ShaderNodeTexChecker")
                tex.inputs["Scale"].default_value = random.uniform(1.0, 50.0)
                tex.inputs["Color1"].default_value = (random.random(), random.random(), random.random(), 1.0)
                tex.inputs["Color2"].default_value = (random.random(), random.random(), random.random(), 1.0)
            elif tex_type == "Voronoi":
                tex = nt.nodes.new("ShaderNodeTexVoronoi")
                tex.inputs["Scale"].default_value = random.uniform(1.0, 50.0)
                # Randomize feature and distance
                tex.feature = random.choice(['F1', 'F2', 'SMOOTH_F1', 'DISTANCE_TO_EDGE', 'N_SPHERE_RADIUS'])
                tex.distance = random.choice(['EUCLIDEAN', 'MANHATTAN', 'CHEBYCHEV', 'MINKOWSKI'])
            elif tex_type == "Magic":
                tex = nt.nodes.new("ShaderNodeTexMagic")
                tex.inputs["Scale"].default_value = random.uniform(1.0, 50.0)
                tex.inputs["Distortion"].default_value = random.uniform(0.0, 5.0)
            
            # We want random RGB static / noise / colors
            # Sometimes we just use the default color outputs of these nodes
            
            # Helper to find a usable output
            def get_output(node, prefs):
                for p in prefs:
                    try:
                        return node.outputs[p]
                    except KeyError:
                        pass
                for o in node.outputs:
                    try:
                        return node.outputs[o.name]
                    except KeyError:
                        pass
                return node.outputs[0]
            
            # Add a color ramp to sometimes make it more random RGB
            if random.random() > 0.5:
                ramp = nt.nodes.new("ShaderNodeValToRGB")
                ramp.color_ramp.elements[0].color = (random.random(), random.random(), random.random(), 1.0)
                ramp.color_ramp.elements[1].color = (random.random(), random.random(), random.random(), 1.0)
                
                # Link texture output to color ramp, then color to diffuse
                tex_out = get_output(tex, ["Fac", "Color", "Distance"])
                nt.links.new(tex_out, ramp.inputs["Fac"])
                    
                nt.links.new(ramp.outputs["Color"], diff.inputs["Color"])
            else:
                tex_out = get_output(tex, ["Color", "Fac", "Distance"])
                nt.links.new(tex_out, diff.inputs["Color"])

            # Clear all old material slots and assign our new one
            o.data.materials.clear()
            o.data.materials.append(mat)
