import bpy
from blendtorch import btb


def main():
    btargs, remainder = btb.parse_blendtorch_args()

    seq = []

    def pre_play(anim):
        seq.extend(["pre_play", anim.frameid])

    def pre_animation(anim):
        seq.extend(["pre_animation", anim.frameid])

    def pre_frame(anim):
        seq.extend(["pre_frame", anim.frameid])

    def post_frame(anim):
        seq.extend(["post_frame", anim.frameid])

    def post_animation(anim):
        seq.extend(["post_animation", anim.frameid])

    def post_play(anim, pub):
        seq.extend(["post_play", anim.frameid])
        pub.publish(seq=seq)

    # Data source: add linger to avoid not sending data upon closing Blender.
    pub = btb.DataPublisher(btargs.btsockets["DATA"], btargs.btid, lingerms=5000)

    anim = btb.AnimationController()
    anim.pre_play.add(pre_play, anim)
    anim.pre_animation.add(pre_animation, anim)
    anim.pre_frame.add(pre_frame, anim)
    anim.post_frame.add(post_frame, anim)
    anim.post_animation.add(post_animation, anim)
    anim.post_play.add(post_play, anim, pub)
    anim.play(frame_range=(1, 3), num_episodes=2, use_animation=not bpy.app.background)


main()
