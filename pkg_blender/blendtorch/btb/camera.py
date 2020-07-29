'''Provides helper functions to deal with Blender cameras.'''
import bpy, bpy_extras
import numpy as np

from . import utils

class Camera:
    def __init__(self, bpy_camera=None, shape=None):
        self.bpy_camera = bpy_camera or bpy.context.scene.camera
        self.shape = shape or Camera.shape_from_bpy()
        self.view_matrix = Camera.view_from_bpy(self.bpy_camera)
        self.proj_matrix = Camera.proj_from_bpy(self.bpy_camera, self.shape)

    def update_view_matrix(self):
        '''Update the view matrix of the camera.'''
        self.view_matrix = Camera.view_from_bpy(self.bpy_camera)

    def update_proj_matrix(self):
        '''Update the projection matrix of the camera.'''
        self.proj_matrix = Camera.proj_from_bpy(self.bpy_camera, self.shape)

    @property
    def type(self):
        '''Returns the Blender type of this camera.'''
        return self.bpy_camera.type

    @property
    def clip_range(self):
        '''Returns the camera clip range.'''
        return (
            self.bpy_camera.data.clip_start, 
            self.bpy_camera.data.clip_end
        )

    @staticmethod
    def shape_from_bpy(bpy_render=None):
        '''Returns the image shape as (HxW) from the given render settings.'''
        render = bpy_render or bpy.context.scene.render
        scale = render.resolution_percentage / 100.0
        shape = (
            int(render.resolution_y * scale),
            int(render.resolution_x * scale)
        )
        return shape

    @staticmethod
    def view_from_bpy(bpy_camera):
        '''Returns 4x4 view matrix from the specified Blender camera.'''
        camera = bpy_camera or bpy.context.scene.camera
        return camera.matrix_world.normalized().inverted()
    
    @staticmethod
    def proj_from_bpy(bpy_camera, shape):
        '''Returns 4x4 projection matrix from the specified Blender camera.'''
        camera = bpy_camera or bpy.context.scene.camera
        shape = shape or Camera.shape_from_bpy()
        return camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(), 
            x=shape[1], y=shape[0]
        )

    def world_to_ndc(self, xyz_world):
        '''Returns normalized device coordinates (NDC) for the given world coordinates.

        Params
        ------
        xyz_world: Nx3 array
            World coordinates given as numpy compatible array.

        Returns
        -------
        xyz_ndc: Nx3 array
            Normalized device coordinates.
        '''

        xyz = np.atleast_2d(xyz_world)
        xyzw = utils.hom(xyz, 1.)
        m = np.asarray(self.proj_matrix @ self.view_matrix)
        ndc = utils.dehom(xyzw @ m.T)
        return ndc


    def ndc_to_linear(self, ndc, origin='upper-left'):
        '''Converts NDC coordinates to pixel and linear depth values
        
        Params
        ------
        ndc: Nx3 array
            Normalized device coordinates.
        origin: str
            Pixel coordinate orgin. Supported values are `upper-left` (OpenCV) and `lower-left` (OpenGL)

        Returns
        -------
        xy: Nx2 array
            Camera pixel coordinates in 
        z: Nx1 array
            Linear depth values.
        '''
        assert origin in ['upper-left', 'lower-left']

        ndc = np.atleast_2d(ndc)
        xyz = (ndc + 1)*0.5 
        if origin == 'upper-left':
            xyz[:, 1] = 1. - xyz[:, 1]

        h,w = self.shape
        xy = xyz[:, :2] * np.array([[w,h]]) 

        cs, ce = self.clip_range
        z = (ce - cs)*xyz[:, -1] + cs

        return xy,z
