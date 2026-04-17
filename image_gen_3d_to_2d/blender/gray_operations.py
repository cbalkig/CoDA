import bpy


class BlenderGrayOps:
    @staticmethod
    def setup(objs, *,
              base_color=(0.5, 0.5, 0.5, 1.0)):
        for o in objs:
            if o.type != "MESH":
                continue

            # Create a brand new material
            mat = bpy.data.materials.new(name="GrayMat")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()

            out = nt.nodes.new("ShaderNodeOutputMaterial")
            emission = nt.nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = base_color
            nt.links.new(emission.outputs["Emission"], out.inputs["Surface"])

            # Clear all old material slots and assign our new one
            o.data.materials.clear()
            o.data.materials.append(mat)

            # Reset every polygon's material index to 0 so that faces which
            # previously referenced slots 1, 2, … (from a multi-material mesh
            # or a joined object) all resolve to the single gray emission mat.
            for poly in o.data.polygons:
                poly.material_index = 0
