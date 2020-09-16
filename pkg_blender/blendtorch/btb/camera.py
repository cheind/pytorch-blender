'''Provides helper functions to deal with Blender cameras.'''
import bpy, bpy_extras
from mathutils import Vector
import numpy as np

from . import utils

class Camera:
    '''Camera related settings and functions.

    An instance of `Camera` is a shallow wrapper around `bpy.types.Camera`
    that provides additional convenience functions as well as intrinsic
    and extrinsic parameters. `Camera` is mainly to be used together with 
    `btb.OffScreenRenderer` to create scene renderings and to convert world
    coordinates to pixel coordinates and linear depth measurements.
    '''


    def __init__(self, bpy_camera=None, shape=None):
        '''Initialize camera object
        
        Params
        ------
        bpy_camera: bpy.types.Camera, None
            Blender camera to attach to. When None, uses the scenes
            default camera.
        shape: tuple, None
            (H,W) of image to create. When None, uses the default 
            render settings.
        '''
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

    def world_to_ndc(self, xyz_world, return_depth=False):
        '''Returns normalized device coordinates (NDC) and optionally linear depth for the given world coordinates.

        Params
        ------
        xyz_world: Nx3 array
            World coordinates given as numpy compatible array.
        return_depth: bool
            Whether or not to return depths w.r.t camera frame.

        Returns
        -------
        ndc: Nx3 array
            Normalized device coordinates.
        z: N array
            Linear depth in camera space. Returned when `return_depth`
            is True.
        '''

        xyz = np.atleast_2d(xyz_world)
        xyzw = utils.hom(xyz, 1.)
        if return_depth:
            xyzw = xyzw @ np.asarray(self.view_matrix).T
            d = -xyzw[:, -2].copy()
            xyzw = xyzw @ np.asarray(self.proj_matrix).T
            return utils.dehom(xyzw), d
        else:
            m = np.asarray(self.proj_matrix @ self.view_matrix)
            return utils.dehom(xyzw @ m.T)


    def ndc_to_pixel(self, ndc, origin='upper-left'):
        '''Converts NDC coordinates to pixel values
        
        Params
        ------
        ndc: Nx3 array
            Normalized device coordinates.
        origin: str
            Pixel coordinate orgin. Supported values are `upper-left` (OpenCV) and `lower-left` (OpenGL)

        Returns
        -------
        xy: Nx2 array
            Camera pixel coordinates
        '''
        assert origin in ['upper-left', 'lower-left']
        h,w = self.shape
        xy = np.atleast_2d(ndc)[:, :2]
        xy = (xy + 1)*0.5 
        if origin == 'upper-left':
            xy[:, 1] = 1. - xy[:, 1]
        return xy * np.array([[w,h]]) 

    def object_to_pixel(self, *objs, return_depth=False):
        '''Convenience composition of `ndc_to_pixel(world_to_ndc(utils.world_coordinates(*objs)))`
        
        Params
        ------
        objs: array of bpy.types.Object
            Collection of objects whose vertices to convert to camera pixel coordinates.
        return_depth: bool
            When True, returns the linear depth in camera space of each coordinate.
            
        Returns
        -------
        xy : Mx2 array
            Concatenated list object vertex coordinates expressed in camera pixels.
        z: array, optional
            Linear depth in camera space when `return_depth==True`
        '''
        if return_depth:
            ndc,z = self.world_to_ndc(utils.world_coordinates(*objs), return_depth=True)
            px = self.ndc_to_pixel(ndc)
            return px, z
        else:
            ndc = self.world_to_ndc(utils.world_coordinates(*objs))
            px = self.ndc_to_pixel(ndc)
            return px
        

    def bbox_object_to_pixel(self, *objs, return_depth=False):
        '''Convenience composition of `ndc_to_pixel(world_to_ndc(utils.bbox_world_coordinates(*objs)))`
        
        Params
        ------
        objs: array of bpy.types.Object
            Collection of objects whose vertices to convert to camera pixel coordinates.
        return_depth: bool
            When True, returns the linear depth in camera space of each coordinate.
            
        Returns
        -------
        xy : Mx2 array
            Concatenated list object vertex coordinates expressed in camera pixels.
        z: array, optional
            Linear depth in camera space when `return_depth==True`
        '''
        if return_depth:
            ndc,z = self.world_to_ndc(utils.bbox_world_coordinates(*objs), return_depth=True)
            px = self.ndc_to_pixel(ndc)
            return px, z
        else:
            ndc = self.world_to_ndc(utils.bbox_world_coordinates(*objs))
            px = self.ndc_to_pixel(ndc)
            return px

    def look_at(self, look_at=None, look_from=None):
        '''Helper function to look at specific location.'''
        if look_from is None:
            look_from = self.bpy_camera.location
        if look_at is None:
            look_at = Vector([0,0,0])

        direction = Vector(look_at) - Vector(look_from)
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.bpy_camera.rotation_euler = rot_quat.to_euler()
        self.bpy_camera.location = look_from
        bpy.context.evaluated_depsgraph_get().update()
        self.update_view_matrix()
