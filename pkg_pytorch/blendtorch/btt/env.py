import zmq

from .constants import DEFAULT_TIMEOUTMS
from .launcher import BlenderLauncher
from .env_rendering import create_renderer

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

    def render(self, mode='human', backend=None):   
        if mode == 'rgb_array' or self.rgb_array is None:
            return self.rgb_array

        if self.viewer is None:
            self.viewer = create_renderer(backend)
        self.viewer.imshow(self.rgb_array)
       
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
            self.viewer.close()
            self.viewer = None
        
from contextlib import contextmanager
@contextmanager
def launch_env(scene, script, **kwargs):
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

# try:
#     import gym
    
#     class OpenAIRemoteGym(object):

#         def __init__(self, **kwargs):
#             self._viewer = rendering.SimpleImageViewer(**kwargs)

#         def imshow(self, rgb):
#             self._viewer.imshow(rgb)

#         def close(self):
#             if self._viewer:
#                 self._viewer.close()
#                 self._viewer = None

#         def __del__(self):
#             self.close()

#     RENDER_BACKENDS['openai'] = OpenAIGymRenderer
# except ImportError as e:
#     pass