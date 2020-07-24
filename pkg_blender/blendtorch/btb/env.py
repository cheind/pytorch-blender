import bpy
import zmq
import sys

from .animation import AnimationController
from .offscreen import OffScreenRenderer
from .constants import DEFAULT_TIMEOUTMS
from . import camera

class BaseEnv:
    '''Environment base class, based on the model of OpenAI Gym.'''

    STATE_INIT = object()
    STATE_RUN = object()
    CMD_RESTART = object()
    CMD_STEP = object()

    def __init__(self, agent):
        self.events = AnimationController()
        self.events.pre_frame.add(self._pre_frame)
        self.events.pre_animation.add(self._pre_animation)
        self.events.post_frame.add(self._post_frame)
        self.agent = agent
        self.ctx = None        
        self.renderer = None
        self.render_every = None
        self.frame_range = None
        self.state = BaseEnv.STATE_INIT

    def run(self, frame_range=None, use_animation=True):
        self.frame_range = AnimationController.setup_frame_range(frame_range)
        self.events.play(
            (self.frame_range[0], 2147483647), # we allow playing the simulation past end.
            num_episodes=-1, 
            use_animation=use_animation, 
            use_offline_render=True)

    def attach_default_renderer(self, every_nth=1):
        self.renderer = OffScreenRenderer(mode='rgb', gamma_coeff=2.2)
        self.renderer.view_matrix = camera.view_matrix()
        self.renderer.proj_matrix = camera.projection_matrix()
        self.render_every = every_nth

    def _pre_frame(self):
        self.ctx['time'] = self.events.frameid
        self.ctx['done'] |= (self.events.frameid >= self.frame_range[1])
        if self.events.frameid > self.frame_range[0]:            
            cmd, action = self.agent(self, **self.ctx)
            if cmd == BaseEnv.CMD_RESTART:
                self._restart()
            elif cmd == BaseEnv.CMD_STEP:
                if action != None:
                    self._env_prepare_step(action)
                    self.ctx['prev_action'] = action
                self.state = BaseEnv.STATE_RUN
            
    def _pre_animation(self):        
        self.state = BaseEnv.STATE_INIT
        self.ctx = {'prev_action':None, 'done':False}
        self._env_reset()

    def _post_frame(self):
        self._render(self.ctx)
        next_ctx = self._env_post_step()
        self.ctx = {**self.ctx, **next_ctx}

    def _render(self, ctx):
        cur, start = self.events.frameid, self.frame_range[0]
        render = bool(
            self.renderer and
            ((cur - start) % self.render_every) == 0
        )
        if render:
            ctx['rgb_array'] = self.renderer.render()

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

    def __init__(self, address, real_time=False, timeoutms=DEFAULT_TIMEOUTMS):        
        self.context = zmq.Context()        
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, timeoutms)
        self.socket.setsockopt(zmq.RCVTIMEO, timeoutms)
        self.socket.bind(address)
        self.real_time = real_time
        self.state = RemoteControlledAgent.STATE_REQ

    def __call__(self, env, **ctx):
        flags = 0
        if self.real_time and (env.state == BaseEnv.STATE_RUN):
            flags = zmq.NOBLOCK

        if self.state == RemoteControlledAgent.STATE_REP:
            try:
                self.socket.send_pyobj(ctx, flags=flags)
                self.state = RemoteControlledAgent.STATE_REQ
            except zmq.error.Again:
                if not self.real_time:
                    raise ValueError('Failed to send to remote agent.')
                return BaseEnv.CMD_STEP, None

        if self.state == RemoteControlledAgent.STATE_REQ:
            try:
                rcv = self.socket.recv_pyobj(flags=flags)
                assert rcv['cmd'] in ['reset', 'step']
                self.state = RemoteControlledAgent.STATE_REP

                if rcv['cmd'] == 'reset':
                    cmd = BaseEnv.CMD_RESTART
                    action = None
                    if env.state == BaseEnv.STATE_INIT:
                        # Already reset
                        cmd, action = self.__call__(env, **ctx)
                elif rcv['cmd'] == 'step':
                    cmd = BaseEnv.CMD_STEP
                    action = rcv['action']
                return cmd, action
            except zmq.error.Again:
                return BaseEnv.CMD_STEP, None
