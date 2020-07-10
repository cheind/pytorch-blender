import numpy as np
import bpy
import logging

from blendtorch import btb

def main():
    args, remainder = btb.parse_blendtorch_args()
    np.random.seed(args.btseed)
    cam = bpy.context.scene.camera

    # Random colors for all cubes
    cubes = list(bpy.data.collections['Cubes'].objects)
    for idx, c in enumerate(cubes):
        mat = bpy.data.materials.new(name=f'random{idx}')
        mat.diffuse_color = np.concatenate((np.random.random(size=3), [1.]))
        c.data.materials.append(mat)
        c.active_material = mat

    def pre_anim(offscreen):
        # Random initial positions
        xyz = np.random.uniform((-3,-3,6),(3,3,12.),size=(len(cubes),3))
        rot = np.random.uniform(-np.pi, np.pi, size=(len(cubes),3))
        for idx, c in enumerate(cubes):
            c.location = xyz[idx]
            c.rotation_euler = rot[idx]
        offscreen.enabled = True
        
    def post_anim(offscreen):
        offscreen.enabled = False

    def post_image(arr, pub): 
        xy = [btb.camera.project_points(c, camera=cam) for c in cubes]   
        pub.publish(
            image=arr, 
            xy=np.concatenate(xy, axis=0),
            frameid=bpy.context.scene.frame_current)

    pub = btb.BlenderOutputChannel(args.bind_address, args.btid)

    off = btb.OffScreenRenderer()
    off.view_matrix = btb.camera.view_matrix()
    off.proj_matrix = btb.camera.projection_matrix()
    off.post_image.add(post_image, pub=pub)

    anim = btb.Controller()
    anim.pre_animation.add(pre_anim, off)
    anim.post_animation.add(post_anim, off)
    anim.play(once=False, startframe=0, endframe=200)

main()