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

    def pre_anim(offscreen):
        offscreen.enabled = True
        
    def post_anim(offscreen):
        offscreen.enabled = False

    def pre_frame():
        mat.diffuse_color = np.concatenate((np.random.random(size=3), [1.]))
        
    def post_image(arr, pub, sub):
        # After an image is ready, we publish data to PyTorch
        pub.publish(
            image=arr, 
            xy=btb.camera.project_points(obj, camera=cam),
            frameid=bpy.context.scene.frame_current)

        # In this demo we check for a message from PyTorch every time
        # we publish. `recv` returns a list of message that
        # have been read.
        rcv = sub.recv(timeoutms=0)
        if rcv is not None:
            print(f'Blender {args.btid} received {rcv}') # Print all messages

    pub = btb.BlenderOutputChannel(args.btoutaddress, args.btid)
    sub = btb.BlenderInputChannel(args.btinaddress)

    off = btb.OffScreenRenderer()
    off.view_matrix = btb.camera.view_matrix()
    off.proj_matrix = btb.camera.projection_matrix()
    off.post_image.add(post_image, pub=pub, sub=sub)

    anim = btb.Controller()
    anim.pre_animation.add(pre_anim, off)
    anim.post_animation.add(post_anim, off)
    anim.pre_frame.add(pre_frame)
    anim.play(once=False, startframe=0, endframe=100)

main()