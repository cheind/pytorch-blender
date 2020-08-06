import bpy
import numpy as np
from blendtorch import btb

import sys
sys.path.append('C:/dev/supershape')
import supershape as sshape

def main():
    btargs, remainder = btb.parse_blendtorch_args()

    SHAPE = (100,100)
    PARAMS = np.array([
        [10, 1, 1, 3, 3, 3],
        [10, 1, 1, 3, 3, 3],
    ], dtype=np.float32)
    cur_params = None    
    STDSCALE = 0.2
    x,y,z = sshape.supercoords(sshape.FLOWER, shape=SHAPE)
    obj = sshape.make_bpy_mesh(x, y, z)
    mid = -1

    def pre_frame(duplex):
        nonlocal mid,  cur_params
        # Randomize cube rotation
        msgs = duplex.recv(timeoutms=0)
        if len(msgs) > 0:
            PARAMS[0,0] = msgs[-1]['m1']
            PARAMS[1,0] = msgs[-1]['m2']
            mid = msgs[-1]['mid']
            print('cur params are now ', PARAMS[:, 0])

        cur_params = PARAMS.copy()
        cur_params[:, 0] += np.random.normal(scale=STDSCALE, size=2).astype(np.float32)
        x,y,z = sshape.supercoords(cur_params, shape=SHAPE)
        sshape.update_bpy_mesh(x,y,z,obj)
        
    def post_frame(off, pub):
        pub.publish(
            image=off.render(), 
            params=cur_params[:, 0],
            mid=mid
        )

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)
    duplex = btb.DuplexChannel(btargs.btsockets['CTRL'])

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb')
    off.set_render_style(shading='SOLID', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame, duplex)
    anim.post_frame.add(post_frame, off, pub)
    anim.play(frame_range=(0,100), num_episodes=-1)

    
main()

