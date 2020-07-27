'''Provides helper functions to deal with Blender cameras.'''
import bpy, bpy_extras
import numpy as np

def image_shape():
    '''Returns resulting image shape as (HxW)'''
    scale = bpy.context.scene.render.resolution_percentage / 100.0
    shape = (
        int(bpy.context.scene.render.resolution_y * scale),
        int(bpy.context.scene.render.resolution_x * scale)
    )
    return shape

def view_matrix(camera=None):
    '''Returns view matrix from the specified camera.'''
    camera = camera or bpy.context.scene.camera
    return camera.matrix_world.inverted()

def projection_matrix(camera=None):
    '''Returns the projection matrix from the specified camera.'''
    camera = camera or bpy.context.scene.camera
    shape = image_shape()
    return camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=shape[1], y=shape[0])

def project_points(obj, camera=None):
    '''Returns 2D pixel projections of all vertices belonging to `obj`.
    
    Params
    ------
    obj: bpy.types.Object
        The object whose vertices to project
    camera: bpy.types.Camera
        The camera to project into. If None, uses the scenes default camera.
    
    Returns
    -------
    xy: Nx2 array
        XY coordinates of each vertex as projected into the camera image.
    '''
    camera = camera or bpy.context.scene.camera
    scene = bpy.context.scene
    mat = obj.matrix_world

    xyz = []
    for v in obj.data.vertices:
        p = bpy_extras.object_utils.world_to_camera_view(scene, camera, mat @ v.co)
        xyz.append(p) # Normalized 2D in W*RS / H*RS

    xyz = np.stack(xyz).astype(np.float32)
    xyz[...,1] = 1. - xyz[...,1] # Blender has origin bottom-left.
    xy = xyz[..., :2]   # normalized xy but unnormalized in z
    
    h,w = image_shape()
    return xy * np.array([[w,h]])