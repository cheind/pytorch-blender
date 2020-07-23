import bpy
import numpy as np
import argparse
from blendtorch import btb

def main():
    btargs, remainder = btb.parse_blendtorch_args()
    script_args = parse_additional_args(remainder)

    cam = bpy.context.scene.camera
    obj = bpy.data.objects["Cube"]
    mat = bpy.data.materials.new(name='cube_random')
    obj.data.materials.append(mat)
    obj.active_material = mat
        
    def pre_frame():
        # Called every time before a frame is processed.
        obj.rotation_euler = np.random.uniform(0,np.pi,size=3)  
        mat.diffuse_color = np.concatenate((np.random.random(size=3), [1.]))
        
    def post_frame(off, pub, anim):
        # Called every after Blender finished processing a frame.
        pub.publish(
            image=off.render(), 
            xy=btb.camera.project_points(obj, camera=cam),
            frameid=anim.frameid
        )

    # Our output channel
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup image rendering
    off = btb.OffScreenRenderer(mode=script_args.mode)
    off.view_matrix = btb.camera.view_matrix()
    off.proj_matrix = btb.camera.projection_matrix()
    off.set_render_style(shading='RENDERED', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, off, pub, anim)    
    anim.play(frame_range=(0,100), num_episodes=-1, use_animation=not script_args.noui)

def parse_additional_args(args):
    '''Parse additional args parsed to this script.
    Used by benchmark.py to configure various parameters of the environment.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-ui-refresh', action='store_true', dest='noui')
    parser.add_argument('--mode', default='rgb')
    return parser.parse_args(args)

main()