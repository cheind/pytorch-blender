import numpy as np
import bpy
from .camera import Camera
from pathlib import Path
import re
from functools import partial
import OpenEXR, Imath
import os

class Renderer:
    '''Provides rendering using Eevee.

    Does not use preview rendering as provided by OffscreenRenderer. This class currently
    requires compositor usage with at least one file output node configured.    
    
    Params
    ------
    camera: btb.Camera, None
        Camera view to be rendered. When None, default camera is used.
    mode: str
        Defines the number of color channels. Either 'RGBA' or 'RGB'
    gamma_coeff: scalar, None
        When not None, applies gamma color correction to the rendered image.
        Blender performs offline rendering in linear color space that when 
        viewed directly appears to be darker than expected. Usually a value
        of 2.2 get's the job done. Defaults to None.
    '''
    
    def __init__(self, camera=None, mode='rgba', gamma_coeff=None, delete_render_files=True):
        assert mode in ['rgba', 'rgb']        
        self.camera = camera or Camera()
        self.gamma_coeff = gamma_coeff
        self.delete_render_files = delete_render_files
        self.channels = 4 if mode=='rgba' else 3
        self.color_image = np.zeros(self.camera.shape + (self.channels,), dtype=np.float32)

        self._scene = bpy.context.scene
        assert self._scene.use_nodes, 'Renderer currently requires compositing nodes.'

        tree = self._scene.node_tree
        outnodes = [n for n in tree.nodes if n.type=='OUTPUT_FILE']
        assert len(outnodes) == 1, 'Renderer requires exactly one fileoutput node'               
        n = outnodes[0]
        assert n.format.file_format == 'OPEN_EXR_MULTILAYER', 'Renderer requires OPEN_EXR_MULTILAYER format.'        
        assert n.format.exr_codec == 'NONE', 'Renderer requires EXR codec None.'
        assert n.format.color_depth == '16', 'Renderer requires half-precision.'
        self._color_depth = Imath.PixelType(Imath.PixelType.HALF)
        
        self._layername = None
        for (inp, slot) in zip(n.inputs, n.layer_slots):
            if inp.type == 'RGBA':
                self._layername = slot.name

        assert self._layername is not None, 'Failed to find RGBA layer.'
        self._outpath_template = n.base_path
        self._outpath_re = re.compile(r'((#)+)')
        
    @property
    def shape(self):
        return self.camera.shape


    def _repl(self, g, fidx):
        len = g.span()[1] - g.span()[0]
        return str(fidx).zfill(len)

    def render(self):
        '''Render the scene and return image as buffer.
        
        Returns
        -------
        image: HxWxD array
            where D is 4 when `mode=='RGBA'` else 3.
        '''
        fidx = self._scene.frame_current

        basepath, cnt = self._outpath_re.subn(partial(self._repl, fidx=fidx), self._outpath_template)
        if cnt == 0:
            basepath = f'{self._outpath_template}{fidx}'
        path = Path(bpy.path.abspath(basepath))
        assert path.exists(), f'Could not find output file {path}'

        file = None
        try:
            file = OpenEXR.InputFile(str(path))
            self.color_image[..., 0] = np.frombuffer(file.channel(f'{self._layername}.R', self._color_depth), dtype=np.float16).reshape(self.shape)
            self.color_image[..., 1] = np.frombuffer(file.channel(f'{self._layername}.G', self._color_depth), dtype=np.float16).reshape(self.shape)
            self.color_image[..., 2] = np.frombuffer(file.channel(f'{self._layername}.B', self._color_depth), dtype=np.float16).reshape(self.shape)
            if self.channels == 4:
                self.color_image[..., 3] = np.frombuffer(file.channel(f'{self._layername}.A', self._color_depth), dtype=np.float16).reshape(self.shape)
        finally:
            if file is not None:
                file.close()
            file = None

        #if self.delete_render_files:
        #    os.remove(path)
        if self.gamma_coeff:
            self._color_correct(self.color_image, self.gamma_coeff)

        return (self.color_image * 255.0).astype(np.uint8)
        
    def _color_correct(self, coeff=2.2):
        ''''Return sRGB image.'''
        mask = (self.color_image[...,:3] <= 0.0031308)
        self.color_image[mask,:3] *= 12.92
        self.color_image[~mask,:3] = 1.055*np.power(self.color_image[~mask,:3], 1/coeff) - 0.055
