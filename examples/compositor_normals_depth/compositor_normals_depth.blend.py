
import blendtorch.btb as btb
import numpy as np
import bpy

SHAPE = (30, 30)
NSHAPES = 70


def main():
    # Update python-path with current blend file directory
    btb.add_scene_dir_to_path()
    import scene_helpers as scene

    def pre_anim(meshes):
        # Called before each animation
        # Randomize supershapes
        for m in meshes:
            scene.update_mesh(m, sshape_res=SHAPE)

    def post_frame(render, pub, animation):
        # After frame
        if anim.frameid == 1:
            imgs = render.render()
            pub.publish(
                normals=imgs['normals'],
                depth=imgs['depth']
            )

    # Parse script arguments passed via blendtorch launcher
    btargs, _ = btb.parse_blendtorch_args()

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
    anim.play(frame_range=(0, 1), num_episodes=-1,
              use_offline_render=False, use_physics=True)


main()
