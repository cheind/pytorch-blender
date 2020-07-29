import bpy
import numpy as np

from blendtorch import btb

def main():
    # Parse script arguments passed via blendtorch
    btargs, remainder = btb.parse_blendtorch_args()

    cam = bpy.context.scene.camera
    cube = bpy.data.objects["Cube"]

    def pre_frame():
        # Randomize cube rotation
        cube.rotation_euler = np.random.uniform(0,np.pi,size=3)  
        
    def post_frame(off, pub, anim, cam):
        # Called every after Blender finished processing a frame.
        image = off.render()
        xy,z = cam.ndc_to_linear(
            cam.world_to_ndc(
                btb.utils.world_coordinates(cube)
            )
        )
        pub.publish(image=image, xy=xy, frameid=anim.frameid)

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb')
    off.set_render_style(shading='RENDERED', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, off, pub, anim, cam)    
    anim.play(frame_range=(0,100), num_episodes=-1)

main()