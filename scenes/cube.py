import numpy as np
import bpy

from blendtorch import blender as btb
  
def main():
    args, remainder = btb.parse_blendtorch_args()

    obj = bpy.data.objects["Cube"]
    randomrot = lambda: np.random.uniform(0,2*np.pi)    
    bpy.app.driver_namespace["randomrot"] = randomrot

    for i in range(3):
        drv = obj.driver_add('rotation_euler', i)
        drv.driver.expression = f'randomrot()'

    def started(offscreen):
        offscreen.enabled = True
        print('started')
        
    def stopped(offscreen):
        offscreen.enabled = False
        print('stopped')
        
    def after_image(arr, pub):    
        pub.publish(image=arr)

    pub = btb.Publisher(args.bind_address, args.btid)

    off = btb.OffScreenRenderer()
    off.set_render_options()
    off.after_image.add(after_image, pub=pub)

    anim = btb.Controller()
    anim.before_animation.add(started, off)
    anim.after_animation.add(stopped, off)
    anim.play(once=False, startframe=0, endframe=10)

main()