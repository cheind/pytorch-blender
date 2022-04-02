import bpy  # noqa
from blendtorch import btb

# See https://github.com/cheind/supershape and this examples readme.
import supershape as sshape


def generate_supershape(msg, shape=(100, 100)):
    for params, shape_id in zip(msg["shape_params"], msg["shape_ids"]):
        yield (params, shape_id, sshape.supercoords(params, shape=shape))


def main():
    btargs, remainder = btb.parse_blendtorch_args()

    uvshape = (100, 100)
    obj = sshape.make_bpy_mesh(uvshape)
    idx = None
    coords = None
    params = None
    gen = None

    def pre_frame(duplex):
        nonlocal gen, params, coords, idx
        msg = duplex.recv(timeoutms=0)
        if msg is not None:
            gen = generate_supershape(msg, shape=uvshape)
        if gen is not None:
            try:
                params, idx, coords = next(gen)
                sshape.update_bpy_mesh(*coords, obj)
            except StopIteration:
                gen = None

    def post_frame(off, pub):
        if gen is not None:
            pub.publish(image=off.render(), shape_id=idx)

    # Data source
    pub = btb.DataPublisher(btargs.btsockets["DATA"], btargs.btid)
    duplex = btb.DuplexChannel(btargs.btsockets["CTRL"], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode="rgb")
    off.set_render_style(shading="SOLID", overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame, duplex)
    anim.post_frame.add(post_frame, off, pub)
    anim.play(frame_range=(0, 10000), num_episodes=-1)


main()
