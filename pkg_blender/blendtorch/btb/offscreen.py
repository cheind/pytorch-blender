import numpy as np
import bpy, gpu, bgl
from OpenGL.GL import glGetTexImage

from .signal import Signal
from . import camera
from .utils import find_first_view3d

class OffScreenRenderer:
    '''Provides off-screen scene rendering.'''
    
    def __init__(self, flip=True, mode='rgba', gamma_coeff=None):
        assert mode in ['rgba', 'rgb']
        self.shape = camera.image_shape()
        self.offscreen = gpu.types.GPUOffScreen(self.shape[1], self.shape[0])
        self.area, self.space, self.region = find_first_view3d()
        self.handle = None
        self.flip = flip
        self.gamma_coeff = gamma_coeff
        self.proj_matrix = camera.projection_matrix()
        self.view_matrix = camera.view_matrix()        
        channels = 4 if mode=='rgba' else 3        
        self.buffer = np.zeros((self.shape[0], self.shape[1], channels), dtype=np.uint8)        
        self.mode = bgl.GL_RGBA if mode=='rgba' else bgl.GL_RGB

    def render(self):
        with self.offscreen.bind():
            self.offscreen.draw_view3d(
                bpy.context.scene,
                bpy.context.view_layer,
                self.space,            
                self.region,
                #bpy.context.space_data,
                #bpy.context.region,
                self.view_matrix,
                self.proj_matrix)
                                
            bgl.glActiveTexture(bgl.GL_TEXTURE0)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.offscreen.color_texture)   

            # np.asarray seems slow, because bgl.buffer does not support the python buffer protocol
            # bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGB, bgl.GL_UNSIGNED_BYTE, self.buffer)
            # https://docs.blender.org/api/blender2.8/gpu.html       
            # That's why we use PyOpenGL at this point instead.     
            glGetTexImage(bgl.GL_TEXTURE_2D, 0, self.mode, bgl.GL_UNSIGNED_BYTE, self.buffer)

        buffer = self.buffer
        if self.flip:
            buffer = np.flipud(buffer)
        if self.gamma_coeff:
            buffer = self._color_correct(buffer, self.gamma_coeff)
        return buffer

    def set_render_style(self, shading='RENDERED', overlays=False):
        self.space.shading.type = shading
        self.space.overlay.show_overlays = overlays

    def _color_correct(self, buffer, coeff=2.2):
        ''''Return sRGB image.'''
        rgb = buffer[...,:3].astype(np.float32) / 255
        rgb = np.uint8(255.0 * rgb**(1/coeff))
        if buffer.shape[-1] == 4:
            return np.concatenate((rgb, buffer[...,3:4]), axis=-1)
        else:
            return rgb
