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

    def pre_anim():
        xyz = np.random.uniform((-3,-3,6),(3,3,12.),size=(len(cubes),3))
        rot = np.random.uniform(-np.pi, np.pi, size=(len(cubes),3))
        for idx, c in enumerate(cubes):
            c.location = xyz[idx]
            c.rotation_euler = rot[idx]

    def post_frame(anim, off, pub): 
        xy = [btb.camera.project_points(c, camera=cam) for c in cubes]   
        pub.publish(
            image=off.render(), 
            xy=np.concatenate(xy, axis=0),
            frameid=anim.frameid)

    pub = btb.DataPublisher(args.btsockets['DATA'], args.btid)

    off = btb.OffScreenRenderer(mode='rgb')
    off.view_matrix = btb.camera.view_matrix()
    off.proj_matrix = btb.camera.projection_matrix()
    off.set_render_style(shading='RENDERED', overlays=False)

    anim = btb.AnimationController()
    anim.pre_animation.add(pre_anim)
    anim.post_frame.add(post_frame, anim, off, pub)
    anim.play(frame_range=(0,100), num_episodes=-1)

main()