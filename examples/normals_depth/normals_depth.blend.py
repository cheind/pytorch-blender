import bpy
import numpy as np

import blendtorch.btb as btb

# Update python-path with current blend file directory,
# so that package `tless` can be found.
import sys
p = bpy.path.abspath("//")
if p not in sys.path:
    sys.path.append(p)
from normals_depth import scene

SHAPE = (30,30)
NSHAPES = 50

def main():
    def pre_anim(meshes):
        # Called before each animation
        # Randomize supershapes
        [scene.update_mesh(m, sshape_res=SHAPE) for m in meshes]

    def post_frame(render, pub, animation):
        # After frame
        if anim.frameid == 2:
            imgs = render.render()
            pub.publish(
                normals=imgs['normals'],
                depth=imgs['depth']
            )

    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()

    # Fetch camera
    cam = bpy.context.scene.camera

    bpy.context.scene.rigidbody_world.time_scale = 100
    bpy.context.scene.rigidbody_world.substeps_per_frame = 300

    # Setup supershapes
    meshes = scene.prepare(NSHAPES, sshape_res=SHAPE)

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
    anim.pre_animation.add(pre_anim, meshes)
    anim.post_frame.add(post_frame, render, pub, anim)
    anim.play(frame_range=(1, 3), num_episodes=-1, use_offline_render=False)


main()
