import bpy
import numpy as np
from blendtorch import btb

import sys
sys.path.append('C:/dev/supershape')
import supershape as sshape

def generate_supershape(msg, shape=(100,100)):
    for p in msg['shape_params']:
        yield p, sshape.supercoords(p, shape=shape)

def main():
    btargs, remainder = btb.parse_blendtorch_args()

    obj = None
    coords = None
    params = None
    gen = None
    
    def pre_frame(duplex):
        nonlocal gen, params, coords, obj        
        msg = duplex.recv(timeoutms=0)
        if msg != None:
            gen = generate_supershape(msg)
        if gen != None:
            try:
                params, coords = next(gen)
                if obj == None:
                    obj = sshape.make_bpy_mesh(*coords)
                else:
                    sshape.update_bpy_mesh(*coords, obj)
            except StopIteration:
                gen = None


    def post_frame(off, pub):
        if gen != None:
            pub.publish(
                image=off.render(), 
                params=params
            )

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)
    duplex = btb.DuplexChannel(btargs.btsockets['CTRL'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb', gamma_coeff=2.2)
    off.set_render_style(shading='SOLID', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame, duplex)
    anim.post_frame.add(post_frame, off, pub)
    anim.play(frame_range=(0,10000), num_episodes=-1)

    
main()

