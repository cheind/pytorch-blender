import bpy
import numpy as np

import blendtorch.btb as btb


def main():
    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()

    cam = bpy.context.scene.camera
    cube = bpy.data.objects["Cube"]

    def pre_frame():
        # Randomize cube rotation
        cube.rotation_euler = np.random.uniform(0, np.pi, size=3)

    def post_frame(off, pub, anim, cam):
        # Called every after Blender finished processing a frame.
        # Will be sent to one of the remote dataset listener connected.
        pub.publish(
            image=off.render(), xy=cam.object_to_pixel(cube), frameid=anim.frameid
        )

    # Data source
    pub = btb.DataPublisher(btargs.btsockets["DATA"], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode="rgb")
    off.set_render_style(shading="RENDERED", overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, off, pub, anim, cam)
    anim.play(frame_range=(0, 100), num_episodes=-1)


main()
