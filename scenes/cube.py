import numpy as np
import bpy
import logging

from blendtorch import btb

def main():
    args, remainder = btb.parse_blendtorch_args()


    cam = bpy.context.scene.camera
    obj = bpy.data.objects["Cube"]
    mat = bpy.data.materials.new(name='random')
    mat.diffuse_color = (1,1,1,1)
    obj.data.materials.append(mat)
    obj.active_material = mat
    
    randomrot = lambda: np.random.uniform(0,2*np.pi)    
    bpy.app.driver_namespace["randomrot"] = randomrot

    for i in range(3):
        drv = obj.driver_add('rotation_euler', i)
        drv.driver.expression = f'randomrot()'

    def started(offscreen):
        offscreen.enabled = True
        
    def stopped(offscreen):
        offscreen.enabled = False

    def before_frame():
        mat.diffuse_color = np.concatenate((np.random.random(size=3), [1.]))
        
    def after_image(arr, pub):    
        pub.publish(image=arr, xy=btb.camera.project_points(obj, camera=cam), frameid=bpy.context.scene.frame_current)

    pub = btb.Publisher(args.bind_address, args.btid)

    off = btb.OffScreenRenderer()
    off.update_perspective(cam)
    off.set_render_options()
    off.after_image.add(after_image, pub=pub)

    anim = btb.Controller()
    anim.before_animation.add(started, off)
    anim.after_animation.add(stopped, off)
    anim.before_frame.add(before_frame)
    anim.play(once=False, startframe=0, endframe=100)

main()