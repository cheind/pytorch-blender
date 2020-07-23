import numpy as np
import bpy
import logging

from blendtorch import btb

def main():
    args, remainder = btb.parse_blendtorch_args()

    cam = bpy.context.scene.camera
    obj = bpy.data.objects["Cube"]
    mat = bpy.data.materials.new(name='cube_random')
    obj.data.materials.append(mat)
    obj.active_material = mat
        
    def pre_frame():
        obj.rotation_euler = np.random.uniform(0,np.pi,size=3)  
        mat.diffuse_color = np.concatenate((np.random.random(size=3), [1.]))
        
    def post_frame(off, pub, anim):
        pub.publish(
            image=off.render(), 
            xy=btb.camera.project_points(obj, camera=cam),
            frameid=anim.frameid
        )

    pub = btb.BlenderOutputChannel(args.btsockets['DATA'], args.btid)

    off = btb.OffScreenRenderer(mode='rgb')
    off.view_matrix = btb.camera.view_matrix()
    off.proj_matrix = btb.camera.projection_matrix()
    off.set_render_style(shading='RENDERED', overlays=False)

    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, off, pub, anim)    
    anim.play(frame_range=(0,100), num_episodes=-1)

main()