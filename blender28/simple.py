import numpy as np
import bpy, bpy_extras, gpu, bgl
from functools import partial
from OpenGL.GL import glGetTexImage
from PIL import Image


class PlayAnimationOnce:
        
    def play(self, startfn=None, finishedfn=None):
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        self.pstop = partial(self.stop, finishedfn=finishedfn)
        bpy.app.handlers.frame_change_pre.append(self.pstop)                
        
        if startfn is not None:
            startfn()
        bpy.ops.screen.animation_play()                
        
                        
    def stop(self, scene, *args, finishedfn=None):
        if bpy.context.scene.frame_current == bpy.context.scene.frame_end:
            bpy.ops.screen.animation_cancel()
            bpy.app.handlers.frame_change_pre.remove(self.pstop)
            if finishedfn is not None:
                finishedfn()        

class OffScreenRenderer:
    def __init__(self, shape=None):        
        if shape is None:
            scale = bpy.context.scene.render.resolution_percentage / 100.0
            shape = (
                int(bpy.context.scene.render.resolution_y * scale),
                int(bpy.context.scene.render.resolution_x * scale)
            )
            
        self.shape = shape        
        self.offscreen = gpu.types.GPUOffScreen(shape[1], shape[0])
        self.camera = bpy.context.scene.camera
        self.buffer = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        self.update_camera()
        self.area, self.space = self.find_view3d()
        
    def update_camera(self):
        self.view_matrix = self.camera.matrix_world.inverted()
        self.proj_matrix = self.camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=self.shape[1], y=self.shape[0])
                
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

        pimg = Image.fromarray(np.flipud(self.buffer))  
        pimg.save(f'c:/tmp/{bpy.context.scene.frame_current:05d}_image.bmp')


    def find_view3d(self):
        areas = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D']
        assert len(areas) > 0
        spaces = [s for s in areas[0].spaces if s.type == 'VIEW_3D']
        assert len(spaces) > 0
        return areas[0], spaces[0]

    def set_render_options(self, shading='RENDERED', overlays=False):
        self.space.shading.type = shading
        self.space.overlay.show_overlays = overlays


off = OffScreenRenderer()
off.set_render_options()

#off.handle = handle


#off.handle = handle

obj = bpy.data.objects["Cube"]
randomrot = lambda: np.random.uniform(0,2*np.pi)    
bpy.app.driver_namespace["randomrot"] = randomrot

for i in range(3):
    drv = obj.driver_add('rotation_euler', i)
    drv.driver.expression = f'randomrot()'

bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 100

def started():
    off.handle = bpy.types.SpaceView3D.draw_handler_add(off.render, (), 'WINDOW', 'POST_PIXEL')
    print('started')
    
def stopped():
    bpy.types.SpaceView3D.draw_handler_remove(off.handle, 'WINDOW')
    print('stopped')

anim = PlayAnimationOnce()
anim.play(started, stopped)

#off.enabled = True
#for i in range(0,10):
#    xy = np.random.uniform(-2,2,size=2)
#    #x = random.randrange(spawn_range[0][0], spawn_range[0][1])
#    #y = random.randrange(spawn_range[1][0], spawn_range[1][1])
#    #print(x)
#    #z = (-0.2)
#    bpy.context.scene.frame_set(i)
#    ob.location = (xy[0], xy[1], 0)
#    ob.keyframe_insert(data_path="location",index=-1)
#    #frame_number += 2
##    off.area.tag_redraw()
#    bpy.context.view_layer.update()
#off.enabled = False
#bpy.types.SpaceView3D.draw_handler_remove(off.handle, 'WINDOW') 