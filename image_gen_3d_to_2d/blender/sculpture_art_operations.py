import bpy


class BlenderSculptureArtOps:
    @staticmethod
    def setup(objs, *,
              base_color=(1.0, 1.0, 1.0, 1.0)):
        for o in objs:
            if o.type != "MESH":
                continue

            # Create a brand new material
            mat = bpy.data.materials.new(name="SculptureMat")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()

            out = nt.nodes.new("ShaderNodeOutputMaterial")
            diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
            diff.inputs["Color"].default_value = base_color
            nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])

            # Clear all old material slots and assign our new one
            o.data.materials.clear()
            o.data.materials.append(mat)
