import bpy
import numpy as np
from blendtorch import btb

import sys
sys.path.append('C:/dev/supershape')
import supershape as sshape

class RandomSupershapeSampler:
    def __init__(self):
        self.shape = (100, 100)        
        self.last_sample = None
        self.type = 'normal'
        self.type_params = {'mean': np.array([5.,5.]), 'cov': np.diag([1.,1.])**2}

    def update_params(self, msg):
        if msg['type'] == 'uniform':
            self.type = 'uniform'
            self.type_params = {'low': msg['low'], 'high': msg['high']}
        elif msg['type'] == 'normal':
            self.type = 'normal'
            self.type_params = {'mean': msg['mean'], 'cov': np.diag(msg['std'])**2}
        elif msg['type'] == 'lognormal':
            self.type = 'lognormal'
            self.type_params = {'mean': msg['mean'], 'std': msg['std']}
        else:
            raise ValueError('unknown type')            

    def sample(self):
        params = np.array([
            [10, 1, 1, 3, 3, 3],
            [10, 1, 1, 3, 3, 3],
        ], dtype=np.float32)

        if self.type == 'normal':
            params[:, 0] = np.random.multivariate_normal(self.type_params['mean'], self.type_params['cov'])
        elif self.type == 'lognormal':
            params[0, 0] = np.random.lognormal(self.type_params['mean'][0], self.type_params['std'][0])
            params[1, 0] = np.random.lognormal(self.type_params['mean'][1], self.type_params['std'][1])
        else:
            params[:, 0] = np.random.uniform(self.type_params['low'], self.type_params['high'], size=2)
        
        self.last_sample = params
        return sshape.supercoords(params, shape=self.shape)


def main():
    btargs, remainder = btb.parse_blendtorch_args()

    configured = False
    sampler = RandomSupershapeSampler()
    x,y,z = sampler.sample()
    obj = sshape.make_bpy_mesh(x, y, z)
    mid = -1

    def pre_frame(duplex):
        nonlocal mid, configured
        # Randomize cube rotation
        msg = duplex.recv(timeoutms=0)
        if msg != None:
            sampler.update_params(msg)
            mid = msg['btmid']
            configured = True

        x,y,z = sampler.sample()
        sshape.update_bpy_mesh(x,y,z,obj)
        
    def post_frame(off, pub):
        if configured:
            pub.publish(
                image=off.render(), 
                params=sampler.last_sample[:, 0],
                mid=mid
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
    anim.play(frame_range=(0,100), num_episodes=-1)

    
main()

