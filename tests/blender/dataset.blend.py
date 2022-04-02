import bpy
import numpy as np
from blendtorch import btb


def main():
    btargs, remainder = btb.parse_blendtorch_args()

    def post_frame(pub, anim):
        pub.publish(frameid=anim.frameid, img=np.zeros((64, 64), dtype=np.uint8))

    # Data source: add linger to avoid not sending data upon closing Blender.
    pub = btb.DataPublisher(btargs.btsockets["DATA"], btargs.btid, lingerms=5000)

    anim = btb.AnimationController()
    anim.post_frame.add(post_frame, pub, anim)
    anim.play(frame_range=(1, 3), num_episodes=-1, use_animation=not bpy.app.background)


main()
