import bpy
import zmq
import sys

from .animation import AnimationController
from .offscreen import OffScreenRenderer
from .constants import DEFAULT_TIMEOUTMS
from .camera import Camera


class BaseEnv:
    '''Abstract base class for environments to be interacted with by agents.

    This class is what `gym.Env` is to OpenAI gym: it defines the basic interface
    required to be implemented by all Blender environments.

    Blender defines a callback based animation system. This also affects `BaseEnv` in that it 
    requires the agent to be given by a callable method having following signature:
        cmd, action = agent(env, **kwargs)
    The arguments passed via **kwargs are at least `obs`, `reward`, `done`. Other variables
    correspond to additional information (`info` dict in OpenAI). The agent is expected to 
    return a command (BaseEnv.CMD_RESTART or BaseEnv.CMD_STEP) with an optional `action` 
    to perform. Note, `action`, `obs`, `reward` and `info` depend on the actual environment 
    implementation.

    Since a callback based agent is unusual to OpenAI users, blendtorch offers a 
    `RemoteControlledAgent`, that communicates with a remotly implemented agent. The remote
    agent can then be implements with the common blocking: `agent.step()`, `agent.restart()`.
    See `blendtorch.btt` for details.

    Each environment inheriting from `BaseEnv`needs to implement the following three methods
     - `BaseEnv._env_reset()` to reset environment state to initial
     - `BaseEnv._env_prepare_step(action) to apply an action in a pre-frame manner.
     - `BaseEnv._env_post_step()` to gather the environment state, reward and other variables
        after the frame has completed (i.e after physics, and animation have computed their values).
    '''

    STATE_INIT = object()
    STATE_RUN = object()
    CMD_RESTART = object()
    CMD_STEP = object()

    def __init__(self, agent):
        '''Initialize the environment.'''
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
        '''Run the environment.

        This hooks with animation system to receive callbacks. The callbacks eventually
        will invoke the actual environments interface methods as described above.

        Params
        ------
        frame_range: tuple, None
            Start and end (inclusive and starts usually at 1 in Blender). When None,
            uses the configured scene frame range.
        use_animation: bool
            Whether to use Blender's non-blocking animation system or a blocking
            variant. Set this to True, when you want to see the agents actions rendered
            live in Blender. When set to False, does not allow Blender UI to refresh, but
            may run at much higher FPS. Consider when training the agent.
        '''
        self.frame_range = AnimationController.setup_frame_range(frame_range)
        self.events.play(
            # we allow playing the simulation past end.
            (self.frame_range[0], 2147483647),
            num_episodes=-1,
            use_animation=use_animation,
            use_offline_render=True)

    def attach_default_renderer(self, every_nth=1):
        '''Attach a default renderer to the environment.

        Convenience function to provide render images for remotely controlled agents (i.e `env.render()`). Uses the default camera perspective for image generation.

        The image rendered will be provided in the `rgb_array` field of the context provided to the agent.

        Params
        ------
        every_nth: int
            Render every nth frame of the simulation.
        '''

        self.renderer = OffScreenRenderer(camera=Camera(), mode='rgb')
        self.render_every = every_nth

    def _pre_frame(self):
        '''Internal pre-frame callback.'''
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
        '''Internal pre-animation callback.'''
        self.state = BaseEnv.STATE_INIT
        self.ctx = {'prev_action': None, 'done': False}
        self._env_reset()

    def _post_frame(self):
        '''Internal post-frame callback.'''
        self._render(self.ctx)
        next_ctx = self._env_post_step()
        self.ctx = {**self.ctx, **next_ctx}

    def _render(self, ctx):
        '''Internal render command.'''
        cur, start = self.events.frameid, self.frame_range[0]
        render = bool(
            self.renderer and
            ((cur - start) % self.render_every) == 0
        )
        if render:
            ctx['rgb_array'] = self.renderer.render()

    def _restart(self):
        '''Restart the environment internally.'''
        self.events.rewind()

    def _env_reset(self):
        '''Reset the environment state. 

        To be implemented by actual environments. Returns nothing.
        '''
        raise NotImplementedError()

    def _env_prepare_step(self, action):
        '''Prepare environment with action.

        Due to the callback system in  Blender, the agents `step` method
        is best split into two distinct function. One that takes the action
        before a frame is animated/rendered and one that collects the new 
        state/reward once the frame has completed. Doing so, allows
        the physics system to take the action before the frame simulation into
        consideration and thus work out the phyisical state at the end of frame.

        In case a remote controlled agent is used, make sure the action is pickle-able.

        Returns nothing.
        '''

        raise NotImplementedError()

    def _env_post_step(self):
        '''Return the environments new state as dict.

        Requires at least the following fields to be set: `obs`, `reward`. You might
        also want to specify `done`. All other fields set will be passed to the agent 
        as well. 

        In case a remote controlled agent is used, make sure all items are pickle-able.

        Returns
        -------
        ddict: dict
            dictionary of key-values describing the new environment state as well as
            any reward and auxilary information.
        '''
        raise NotImplementedError()


class RemoteControlledAgent:
    '''Agent implementation that receives commands from a remote peer.

    Uses a request(remote-agent)/reply(self) pattern to model a [blocking] 
    service call. The agent is expected to initiate a request using a dictionary:
     - `cmd` field set either to `'reset'` or `'step'`.
     - `action` field set when `cmd=='step'`.
    The remote agent will then be passed a dictionary response that contains
    all kwargs passed from the environment to `RemoteControlledAgent`.

    Per default, request/response pairs will eventually block Blender. That allows
    the remote agent to process each frame of the simulation, independent of the time
    it takes to generate an answer. However, this class also supports a special
    `real_time` flag, in which case the environment continues the simulation. Once an
    agent request arrives, it will be applied to the current simulation time.

    Params
    ------
    address: str
        ZMQ remote address to bind to.
    real_time: bool
        Whether or not to continue simulation while waiting for a 
        new remote agent request. Default False.
    timeoutms: int
        Default timeout in milliseconds to wait for new agent requests,
        only applies when `real_time=True`.
    '''
    STATE_REQ = 0
    STATE_REP = 1

    def __init__(self, address, real_time=False, timeoutms=DEFAULT_TIMEOUTMS):
        '''Initialize the remote controlled agent.'''
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, timeoutms)
        self.socket.setsockopt(zmq.RCVTIMEO, timeoutms)
        self.socket.bind(address)
        self.real_time = real_time
        self.state = RemoteControlledAgent.STATE_REQ

    def __call__(self, env, **ctx):
        '''Process agent environment callback.'''
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
