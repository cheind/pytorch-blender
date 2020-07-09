import numpy as np
import bpy, gpu, bgl
from OpenGL.GL import glGetTexImage

from .signal import Signal
from . import camera

class OffScreenRenderer:
    '''Provides off-screen scene rendering.'''
    
    def __init__(self, flip=True):
        self.shape = camera.image_shape()
        self.offscreen = gpu.types.GPUOffScreen(self.shape[1], self.shape[0])
        self.buffer = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.uint8)
        self.area, self.space = self.find_view3d()
        self.handle = None
        self.flip = flip        
        self.proj_matrix = camera.projection_matrix()
        self.view_matrix = camera.view_matrix()
        self.post_image = Signal()
        self.set_render_options()

    def render(self):
        if not bpy.context.space_data == self.space:
            return

        self.offscreen.draw_view3d(
            bpy.context.scene,
            bpy.context.view_layer,
            bpy.context.space_data,
            bpy.context.region,
            self.view_matrix,
            self.proj_matrix)
                            
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.offscreen.color_texture)   

        # np.asarray seems slow, because bgl.buffer does not support the python buffer protocol
        # bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGB, bgl.GL_UNSIGNED_BYTE, self.buffer)
        # https://docs.blender.org/api/blender2.8/gpu.html       
        # That's why we use PyOpenGL at this point instead.     
        glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, self.buffer)

        buffer = self.buffer
        if self.flip:
            buffer = np.flipud(buffer)

        self.post_image.invoke(buffer)

    def find_view3d(self):
        areas = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D']
        assert len(areas) > 0
        spaces = [s for s in areas[0].spaces if s.type == 'VIEW_3D']
        assert len(spaces) > 0
        return areas[0], spaces[0]

    def set_render_options(self, shading='RENDERED', overlays=False):
        self.space.shading.type = shading
        self.space.overlay.show_overlays = overlays
        
    @property
    def enabled(self):
        return self.handle != None
    
    @enabled.setter 
    def enabled(self, toggle):
        if toggle and self.handle is None:
            self.handle = bpy.types.SpaceView3D.draw_handler_add(self.render, (), 'WINDOW', 'POST_PIXEL')
        elif not toggle and self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
            self.handle = None
