import bpy

def image_shape():
    '''Returns resulting image shape as (HxW)'''
    scale = bpy.context.scene.render.resolution_percentage / 100.0
    shape = (
        int(bpy.context.scene.render.resolution_y * scale),
        int(bpy.context.scene.render.resolution_x * scale)
    )
    return shape

def view_projection_matrix(camera=None):    
    '''Returns view and projection matrix from given camera.'''
    camera = camera or bpy.context.scene.camera
    shape = image_shape()
    view_matrix = camera.matrix_world.inverted()
    proj_matrix = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=shape[1], y=shape[0])
    return view_matrix, proj_matrix

def project_points(obj, camera=None):
    # https://docs.blender.org/api/current/bpy_extras.object_utils.html
    # https://github.com/dfelinto/blender/blob/master/release/scripts/modules/bpy_extras/object_utils.py
    '''Get 2D projection coordinates object's vertex coordinates.'''
    camera = camera or bpy.context.scene.camera
    scene = bpy.context.scene
    data = obj.data
    mat = obj.matrix_world

    verts = data.vertices
    if ids is not None:
        verts = [data.vertices[i] for i in ids]

    xyz = []
    for v in verts:
        p = bpy_extras.object_utils.world_to_camera_view(scene, camera, mat @ v.co)
        xyz.append(p) # Normalized 2D in W*RS / H*RS

    xyz = np.stack(xyz).astype(np.float32)
    xyz[...,1] = 1. - xyz[...,1] # Blender has origin bottom-left.
    return xyz # normalized xy but unnormalized in z

def normalized_keypoints(obj):
    xyz = get_pixels(obj, ids=obj['keypoints'])    
    mask = np.logical_and((xyz > 0).all(1), (xyz[:, :2] < 1).all(1))       
    return xyz, mask
