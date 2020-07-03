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