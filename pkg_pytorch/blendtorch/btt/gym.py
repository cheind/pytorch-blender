import zmq

from .constants import DEFAULT_TIMEOUTMS
from .launcher import BlenderLauncher
import matplotlib.pyplot as plt

class RemoteEnv:
    def __init__(self, address, timeoutms=DEFAULT_TIMEOUTMS):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.timeoutms = timeoutms
        self.rgb_array = None
        self.viewer = None

    def reset(self):
        ddict = self._reqrep('reset', action=None)
        self.rgb_array = ddict.pop('rgb_array', None)
        return ddict.pop('obs'), ddict

    def step(self, action):
        ddict = self._reqrep('step', action=action)        
        obs = ddict.pop('obs')
        r = ddict.pop('reward')
        done = ddict.pop('done')        
        self.rgb_array = ddict.pop('rgb_array', None)
        return obs, r, done, ddict

    def render(self):        
        if self.rgb_array is None:
            return

        if self.viewer is None:                                    
            fig, ax = plt.subplots(1,1)
            img = ax.imshow(self.rgb_array)
            plt.show(block=False)
            fig.canvas.draw()
            self.viewer = (fig,ax,img)
        else:
            fig,ax,img = self.viewer
            img.set_data(self.rgb_array)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
       
    def _reqrep(self, cmd, action=None):
        self.socket.send_pyobj((cmd, action))
        
        socks = dict(self.poller.poll(self.timeoutms))
        assert self.socket in socks, 'No response within timeout interval.'
        return self.socket.recv_pyobj()
        
        obs = data.pop('obs', None)
        r = data.pop('reward', 0.)
        done = data.pop('done', False)        
        return obs, r, done, data

    def close(self):
        if self.viewer:
            fig,ax,img = self.viewer
            plt.close(fig)
            self.viewer = None
        
from contextlib import contextmanager
@contextmanager
def launch_blender_env(scene, script, **kwargs):
    env = None
    try:
        additional_args = []
        for k,v in kwargs.items():
            additional_args.extend([f'--{k}',str(v)])
        
        launcher_args = dict(
            scene=scene, 
            script=script, 
            num_instances=1, 
            named_sockets=['GYM'],
            instance_args=[additional_args]
        )
        with BlenderLauncher(**launcher_args) as bl:            
            env = RemoteEnv(bl.launch_info.addresses['GYM'][0])
            yield env
    finally:
        if env:
            env.close()