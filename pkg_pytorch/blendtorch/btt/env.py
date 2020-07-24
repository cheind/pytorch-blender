import zmq

from .constants import DEFAULT_TIMEOUTMS
from .launcher import BlenderLauncher
from .env_rendering import create_renderer

class RemoteEnv:
    def __init__(self, address, timeoutms=DEFAULT_TIMEOUTMS):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, timeoutms*10)
        self.socket.setsockopt(zmq.RCVTIMEO, timeoutms)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.socket.connect(address)
        self.env_time = None
        self.rgb_array = None
        self.viewer = None

    def reset(self):
        ddict = self._reqrep(cmd='reset')
        self.rgb_array = ddict.pop('rgb_array', None)
        return ddict.pop('obs'), ddict

    def step(self, action):
        ddict = self._reqrep(cmd='step', action=action)
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
       
    def _reqrep(self, **send_kwargs):
        try:
            ext = {**send_kwargs, 'time':self.env_time}
            self.socket.send_pyobj(ext)
        except zmq.error.Again:
            raise ValueError('Failed to send to remote environment') from None
        
        try:
            ddict =  self.socket.recv_pyobj()
            self.env_time = ddict['time']
            return ddict
        except zmq.error.Again:
            raise ValueError('Failed to receive from remote environment') from None

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
            k = k.replace('_', '-')
            if isinstance(v, bool):
                if v:
                    additional_args.append(f'--{k}')
                else:
                    additional_args.append(f'--no-{k}')
            else:
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

try:
    import gym
    from contextlib import ExitStack
    
    class OpenAIRemoteEnv(gym.Env):
        metadata = {'render.modes': ['rgb_array', 'human']}

        def __init__(self,  version='0.0.1'):
            self.__version__ = version
            self._es = ExitStack()
            self._env = None

        def launch(self, scene, script, **kwargs):
            assert not self._env, 'Environment already running.'
            self._env = self._es.enter_context(
                launch_env(
                    scene=scene,
                    script=script,
                    **kwargs
                )
            )

        def step(self, action):
            assert self._env, 'Environment not running.'
            obs, reward, done, info = self._env.step(action)
            return obs, reward, done, info

        def reset(self):
            assert self._env, 'Environment not running.'
            obs, info = self._env.reset()
            return obs
            
        def seed(self, seed):
            raise NotImplementedError()

        def render(self, mode='human'):
            assert self._env, 'Environment not running.'
            return self._env.render(mode=mode)

        @property
        def env_time(self):
            return self._env.env_time

        def close(self):   
            if self._es:     
                self._es.close()
                self._es = None
                self._env = None

        def __del__(self):
            self.close()

except ImportError as e:
    pass