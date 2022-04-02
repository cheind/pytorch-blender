import bpy


def create_normal_material(name):
    """Returns a surface material to render camera-space normals.

    The normal vectors range (-1,1) is transformed to match the color range (0,1) as follows:
        c = n*(0.5,0.5,-0.5) + 0.5
    To recover the orginal normal, thus
        n = (c - 0.5)/(0.5,0.5,-0.5)
    where multiplication and division is component-wise.

    Params
    ------
    name : str
        Name of matrial

    Returns
    -------
    mat : Material
        Blender material
    """
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = m.node_tree
    for n in t.nodes:
        t.nodes.remove(n)

    out = t.nodes.new(type="ShaderNodeOutputMaterial")
    geo = t.nodes.new(type="ShaderNodeNewGeometry")
    vt = t.nodes.new(type="ShaderNodeVectorTransform")
    mul = t.nodes.new(type="ShaderNodeVectorMath")
    add = t.nodes.new(type="ShaderNodeVectorMath")

    # Transform normal from world to camera
    vt.convert_from = "WORLD"
    vt.convert_to = "CAMERA"
    vt.vector_type = "VECTOR"
    t.links.new(geo.outputs["Normal"], vt.inputs["Vector"])

    # Shift and scale to [0..1] range required for colors
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (0.5, 0.5, -0.5)
    t.links.new(vt.outputs["Vector"], mul.inputs[0])

    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    t.links.new(mul.outputs["Vector"], add.inputs[0])

    # Use the output vector as color of the surface
    t.links.new(add.outputs["Vector"], out.inputs["Surface"])

    return m
