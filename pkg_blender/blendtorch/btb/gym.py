import bpy
import zmq
import sys

from .animation import AnimationController
from .constants import DEFAULT_TIMEOUTMS

class BaseEnv:
    '''Environment base class, based on the model of OpenAI Gym.'''

    def __init__(self, agent, frame_range=None, use_animation=True, use_offline_render=True, use_physics=True):
        self.events = AnimationController()
        self.events.pre_frame.add(self._pre_frame)
        self.events.pre_animation.add(self._pre_animation)
        self.events.post_frame.add(self._post_frame)
        self.agent = agent
        self.ctx = None
        self.frame_range = AnimationController.setup_frame_range(frame_range)
        self.events.play(
            (self.frame_range[0], 2147483647), # we allow playing the simulation past end.
            num_episodes=-1, 
            use_animation=use_animation, 
            use_offline_render=use_offline_render,
            use_physics=use_physics)

    def _pre_frame(self):
        self.ctx['time'] = self.events.frameid
        self.ctx['done'] |= (self.events.frameid >= self.frame_range[1])
        if self.events.frameid > self.frame_range[0]:            
            action = self.agent(**self.ctx)
            if action == None:
                self._restart()
            else:
                self._env_prepare_step(action)
                self.ctx['prev_action'] = action        
            
    def _pre_animation(self):        
        self.ctx = {'prev_action':None, 'done':False}
        self._env_reset()

    def _post_frame(self):        
        next_ctx = self._env_post_step()            
        self.ctx = {**self.ctx, **next_ctx}

    def _restart(self):
        self.events.rewind()

    def _env_reset(self):
        raise NotImplementedError()
        
    def _env_prepare_step(self, action):
        raise NotImplementedError()

    def _env_post_step(self):
        raise NotImplementedError()
        
class RemoteControlledAgent:
    STATE_REQ = 0
    STATE_REP = 1

    def __init__(self, address, timeoutms=DEFAULT_TIMEOUTMS):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(address)
        self.state = RemoteControlledAgent.STATE_REQ
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.timeoutms = timeoutms

    def __call__(self, prev_action=None, **ctx):
        if self.state == RemoteControlledAgent.STATE_REP:
            self.socket.send_pyobj(ctx)
            self.state = RemoteControlledAgent.STATE_REQ

        socks = dict(self.poller.poll(self.timeoutms))
        assert self.socket in socks, 'No response within timeout interval.'
        
        cmd, action = self.socket.recv_pyobj()
        assert cmd in ['reset', 'step']
        self.state = RemoteControlledAgent.STATE_REP

        if cmd == 'reset':
            action = None
            if prev_action is None:
                # Already reset
                action = self.__call__(**ctx)
        return action
        

