import bpy
import zmq

from .animation import AnimationController


class AgentContext:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.time = 0
        self.episode = 1
        self.action = None
        self.reward = None
        self.obs = None
        self.done = False

    def __repr__(self):
        return f'AgentContext(action={self.action}, reward={self.reward}, obs={self.obs}, done={self.done}, time={self.time}, episode={self.episode}>'

    def __str__(self):
        return self.__repr__()
    

class BaseEnv:
    '''Environment base class, based on the model of OpenAI Gym.'''

    def __init__(self, agent, frame_range=None):
        self.events = AnimationController()
        self.events.pre_frame.add(self._pre_frame)
        self.events.pre_animation.add(self._pre_animation)
        self.events.post_frame.add(self._post_frame)
        self.agent = agent
        self.agent_context = AgentContext(self)
        self.frame_range = frame_range
        self.events.play(self.frame_range, num_episodes=-1)
        
        # self.context = zmq.Context()
        # self.socket = context.socket(zmq.REP)
        # self.socket.connect(address)
        # self.poller = zmq.Poller()
        # self.poller.register(self.socket, zmq.POLLIN)
        # self.state = 'IDLE'

    def _pre_frame(self):
        if self.events.frameid > self.frame_range[0]:
            self.agent_context.time = self.events.frameid
            action = self.agent(self.agent_context)
            if action == None:
                self._restart()
            else:
                self._env_prepare_step(action, self.agent_context)
                self.agent_context.action = action
            
    def _pre_animation(self):        
        self._env_reset(self.agent_context)

    def _post_frame(self):        
        self._env_post_step(self.agent_context)

    def _restart(self):
        self.events.reset()

    def _env_reset(self, ctx):
        eps = ctx.episode
        ctx.reset()
        ctx.episode = eps + 1
        
    def _env_prepare_step(self, action, ctx):
        raise NotImplementedError()

    def _env_post_step(self, ctx):
        raise NotImplementedError()
        
class RemoteControlledAgent:
    def __init__(self, address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(address)
        self.state = 'AWAIT_REQUEST'

    def __call__(self, ctx):
        if self.state == 'SEND_RESPONSE':
            self.socket.send_pyobj((ctx.obs, ctx.reward, ctx.done))
            self.state = 'AWAIT_REQUEST'
        
        cmd, action = self.socket.recv_pyobj()
        assert cmd in ['reset', 'step']
        self.state = 'SEND_RESPONSE'

        if cmd == 'reset':
            action = None
            if ctx.action is None:
                # Already reset
                action = self.__call__(ctx)        
        return action

        # self.poller = zmq.Poller()
        # self.poller.register(self.socket, zmq.POLLIN)
        # self.state = 'IDLE'
        

