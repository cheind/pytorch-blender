import numpy as np
import bpy, gpu, bgl
from functools import partial
from OpenGL.GL import glGetTexImage
from PIL import Image
  
from blendtorch.blender.controller import Controller
from blendtorch.blender.offscreen import OffScreenRenderer

obj = bpy.data.objects["Cube"]
randomrot = lambda: np.random.uniform(0,2*np.pi)    
bpy.app.driver_namespace["randomrot"] = randomrot

for i in range(3):
    drv = obj.driver_add('rotation_euler', i)
    drv.driver.expression = f'randomrot()'

bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 10

def started(offscreen):
    offscreen.enabled = True
    print('started')
    
def stopped(offscreen):
    offscreen.enabled = False
    print('stopped')

def after_image(arr):    
    pimg = Image.fromarray(np.flipud(arr))  
    pimg.save(f'c:/tmp/{bpy.context.scene.frame_current:05d}_image.bmp')

off = OffScreenRenderer()
off.set_render_options()
off.after_image.add(after_image)

anim = Controller()
anim.before_animation.add(started, off)
anim.after_animation.add(stopped, off)
anim.play_once()

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