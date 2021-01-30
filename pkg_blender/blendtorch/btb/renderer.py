from dataclasses import dataclass
import bpy
import re
import os
import numpy as np
from functools import partial
from itertools import groupby
from pathlib import Path
import minexr
from collections import defaultdict

from .camera import Camera


class Renderer:
    '''Provides rendering using Eevee.

    Does not use preview rendering as provided by OffscreenRenderer. This class currently requires compositor usage with at least one file output node configured.    

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

    def __init__(self, btid, camera=None, mode='rgba', gamma_coeff=None, delete_render_files=True):
        assert mode in ['rgba', 'rgb']

        self.btid = btid
        self.camera = camera or Camera()
        self.gamma_coeff = gamma_coeff
        self.delete_render_files = delete_render_files
        self.channels = 4 if mode == 'rgba' else 3

        self._scene = bpy.context.scene
        print(bpy.context.scene.use_nodes)
        assert self._scene.use_nodes, 'Renderer currently requires compositing nodes.'

        tree = self._scene.node_tree
        outnodes = [n for n in tree.nodes if n.type == 'OUTPUT_FILE']
        assert len(outnodes) == 1, 'Renderer requires exactly one fileoutput node'
        n = outnodes[0]
        assert n.format.file_format == 'OPEN_EXR_MULTILAYER', 'Renderer requires OPEN_EXR_MULTILAYER format.'
        assert n.format.exr_codec == 'NONE', 'Renderer requires EXR codec None.'
        assert n.format.color_depth == '16', 'Renderer requires half-precision.'

        layer = None
        for (inp, slot) in zip(n.inputs, n.layer_slots):
            if inp.type == 'RGBA':
                layer = slot.name
        assert layer is not None, 'Failed to find RGBA layer.'

        if self.channels == 4:
            self._exr_channels = [f'{layer}.{c}' for c in ['R', 'G', 'B', 'A']]
        else:
            self._exr_channels = [f'{layer}.{c}' for c in ['R', 'G', 'B']]

        path = Path(bpy.path.abspath(n.base_path))
        path = path.parent / f'{path.stem}_{btid:02d}'
        path = path.with_suffix('.exr')
        n.base_path = str(path)
        self._outpath_template = str(path)
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
        bpy.ops.render.render(
            animation=False, write_still=False, use_viewport=True)

        basepath, cnt = self._outpath_re.subn(
            partial(self._repl, fidx=fidx), self._outpath_template)
        if cnt == 0:
            basepath = f'{self._outpath_template}{fidx}'
        path = Path(bpy.path.abspath(basepath))
        assert path.exists(), f'Could not find output file {path}'

        with open(path, 'rb') as fp:
            reader = minexr.load(fp)
            rgba = reader.select(self._exr_channels).astype(np.float32)

        if self.delete_render_files:
            os.remove(path)
        if self.gamma_coeff:
            self._color_correct(rgba, coeff=self.gamma_coeff)

        return (rgba * 255.0).astype(np.uint8)

    def _color_correct(self, img, coeff=2.2):
        ''''Return sRGB image.'''
        img[..., :3] = img[..., :3]**(1/coeff)


@dataclass
class CompositeSelection:
    key: str
    node: str
    slot: str
    channels: str


@dataclass
class _EXRSelection:
    key: str
    channels: list


class CompositeRenderer:
    '''Provides composite rendering support for EEVEE.

    Does not use preview rendering as provided by OffscreenRenderer. This class currently requires compositor usage with at least one file output node configured.    

    Params
    ------
    slots: Iterable[SlotSelection]
        Slots to be selected and returned by `render`.        
    camera: btb.Camera, None
        Camera view to be rendered. When None, default camera is used.
    delete_render_files: bool, optional
        Delete intermediate render files. Defaults to true.
    btid: int, optional
        Blendtorch worker index. Required to avoid file collisions for multiple workers.
    '''

    def __init__(self, slots, camera=None, delete_render_files=True, btid=1):
        self.btid = btid
        self.camera = camera or Camera()
        self.delete_render_files = delete_render_files
        self._scene = bpy.context.scene
        assert self._scene.use_nodes, 'CompositeRenderer requires compositing nodes'
        assert len(slots) > 0, 'No slots selected'

        self.outnodes = self._find_output_nodes()
        self.outnodes = self._update_output_paths(self.outnodes)
        self._outpath_re = re.compile(r'((#)+)')

        def node_from_name(name):
            nodes = [n for n in self.outnodes if n.name == name]
            assert len(nodes) > 0, f'Could not find output node {name}'
            return nodes[0]

        def exr_from_slots(slots):
            exr = []
            for s in slots:
                channels = [f'{s.slot}.{c}' for c in s.channels]
                exr.append(_EXRSelection(s.key, channels))
            return exr

        self.mapped_slots = {
            node_from_name(k): exr_from_slots(g)
            for k, g in groupby(slots, key=lambda s: s.node)
        }

    def _find_output_nodes(self):
        tree = self._scene.node_tree
        outnodes = [n for n in tree.nodes if n.type == 'OUTPUT_FILE']

        def is_compatible(n):
            return (
                n.format.file_format == 'OPEN_EXR_MULTILAYER' and
                n.format.exr_codec == 'NONE'  # and
                #n.format.color_depth == '16'
            )

        outnodes = [n for n in outnodes if is_compatible(n)]
        assert len(outnodes) > 0, 'Could not find a compatible output filenode'
        return outnodes

    def _update_output_paths(self, outnodes):
        for n in outnodes:
            path = Path(bpy.path.abspath(n.base_path))
            path = path.parent / f'{path.stem}_{self.btid:02d}'
            path = path.with_suffix('.exr')
            n.base_path = str(path)
        return outnodes

    def _actual_path(self, fidx, base_path):
        def _repl(g, fidx):
            len = g.span()[1] - g.span()[0]
            return str(fidx).zfill(len)

        newpath, cnt = self._outpath_re.subn(
            partial(_repl, fidx=fidx),
            base_path)
        if cnt == 0:
            newpath = f'{base_path}{fidx}'
        path = Path(bpy.path.abspath(newpath))
        assert path.exists(), f'Could not find output file {path}'
        return path

    def render(self):
        '''Render the scene and return image as buffer.

        Returns
        -------
        image: HxWxD array
            where D is 4 when `mode=='RGBA'` else 3.
        '''

        fidx = self._scene.frame_current
        bpy.ops.render.render(
            animation=False,
            write_still=False,
            use_viewport=True)

        key_data = {}
        for node, exrsel in self.mapped_slots.items():
            path = self._actual_path(fidx, node.base_path)
            with open(path, 'rb') as fp:
                reader = minexr.load(fp)
                # print(reader.attrs)
                for exr in exrsel:
                    data = reader.select(exr.channels).astype(np.float32)
                    key_data[exr.key] = data
            if self.delete_render_files:
                os.remove(path)

        return key_data

    # def _color_correct(self, img, coeff=2.2):
    #     ''''Return sRGB image.'''
    #     img[..., :3] = img[..., :3]**(1/coeff)
