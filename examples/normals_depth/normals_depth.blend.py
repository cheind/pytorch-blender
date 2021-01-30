import bpy
import numpy as np

import blendtorch.btb as btb


def main():
    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()

    cam = bpy.context.scene.camera

    def pre_frame():
        # Randomize cube rotation
        #cube.rotation_euler = np.random.uniform(0, np.pi, size=3)
        pass

    def post_frame(render, anim, cam):
        # Called every after Blender finished processing a frame.
        # Will be sent to one of the remote dataset listener connected.
        imgs = render.render()
        pub.publish(
            normals=imgs['normals'],
            depth=imgs['depth']
        )

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    render = btb.CompositeRenderer(
        [
            btb.CompositeSelection('normals', 'Out1', 'Normals', 'RGB'),
            btb.CompositeSelection('depth', 'Out1', 'Depth', 'V'),
        ],
        btid=btargs.btid,
        camera=cam,
    )

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, render, anim, cam)
    anim.play(frame_range=(0, 1), num_episodes=-1, use_offline_render=False)


main()
