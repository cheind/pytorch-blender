from contextlib import contextmanager
import zmq

from .constants import DEFAULT_TIMEOUTMS
from .launcher import BlenderLauncher
from .env_rendering import create_renderer
from . import colors


class RemoteEnv:
    """Communicate with a remote Blender environment.

    This sets up a communication channel with a remote Blender environment.
    Its counterpart on Blender is usually a `btb.RemoteControlledAgent`.

    `RemoteEnv` already provides the usual `step()` and `reset()` methods
    that block the caller until the remote call returns. However, it does
    not manage launching the remote Environment. For this reason we provide
    `launch_env` below.

    To provide OpenAI gym compatible environments, one usually inherits
    from `btb.env.OpenAIRemoteEnv`.

    By default, the simulation time of the remote environment only advances
    when the agent issues a command (step, reset). However, one may configure
    the remote environment in real-time mode, in which case the simulation time
    advances independently of the agent's commands.

    Params
    ------
    address: str
        ZMQ endpoint to connect to.
    timeoutms: int
        Receive timeout before raising an error.
    """

    def __init__(self, address, timeoutms=DEFAULT_TIMEOUTMS):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, timeoutms * 10)
        self.socket.setsockopt(zmq.RCVTIMEO, timeoutms)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.socket.connect(address)
        self.env_time = None
        self.rgb_array = None
        self.viewer = None

    def reset(self):
        """Reset the remote environment.

        Returns
        -------
        obs: object
            Initial observation
        info: dict
            Addition information provided by the remote
            environment.
        """
        ddict = self._reqrep(cmd="reset")
        self.rgb_array = ddict.pop("rgb_array", None)
        return ddict.pop("obs"), ddict

    def step(self, action):
        """Advance the remote environment by providing an action.

        Params
        ------
        action: object
            Action to apply

        Returns
        -------
        obs: object
            New observation
        reward: float
            Received reward
        done: bool
            Whether or not the environment simulation finished
        info: dict
            Additional information provided by the environment.
        """
        ddict = self._reqrep(cmd="step", action=action)
        obs = ddict.pop("obs")
        r = ddict.pop("reward")
        done = ddict.pop("done")
        self.rgb_array = ddict.pop("rgb_array", None)
        return obs, r, done, ddict

    def render(self, mode="human", backend=None, gamma_coeff=2.2):
        """Render the current remote environment state.

        We consider Blender itself the visualization of the environment
        state. By calling this method a 2D render image of the environment
        will be shown, if the remote environment configured a suitable renderer.

        Params
        ------
        mode: str
            Either 'human' or 'rgb_array'
        backend: str, None
            Which backend to use to visualize the image. When None,
            automatically chosen by blendtorch.
        gamma_coeff: scalar
            Gamma correction coeff before visualizing image. Does not
            affect the returned rgb array when mode is `rgb_array` which
            remains in linear color space. Defaults to 2.2
        """

        if mode == "rgb_array" or self.rgb_array is None:
            return self.rgb_array

        if self.viewer is None:
            self.viewer = create_renderer(backend)
        self.viewer.imshow(colors.gamma(self.rgb_array, gamma_coeff))

    def _reqrep(self, **send_kwargs):
        """Convenience request-reply method."""
        try:
            ext = {**send_kwargs, "time": self.env_time}
            self.socket.send_pyobj(ext)
        except zmq.error.Again:
            raise ValueError("Failed to send to remote environment") from None

        try:
            ddict = self.socket.recv_pyobj()
            self.env_time = ddict["time"]
            return ddict
        except zmq.error.Again:
            raise ValueError("Failed to receive from remote environment") from None

    def close(self):
        """Close the environment."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.socket:
            self.socket.close()
            self.socket = None


@contextmanager
def launch_env(scene, script, background=False, **kwargs):
    """Launch a remote environment wrapped in a context manager.

    Params
    ------
    scene: path, str
        Blender scene file
    script: path, str
        Python script containing environment implementation.
    background: bool
        Whether or not this environment can run in Blender background mode.
        Defaults to False.
    kwargs: dict
        Any other arguments passed as command-line arguments
        to the remote environment. Note by default a <key,value>
        entry will be converted to `--key str(value)`. Boolean values
        will be converted to switches as follows `--key` or `--no-key`.
        Note that underlines will be converted to dashes as usual with
        command-line arguments and argparse.

    Yields
    ------
    env: `btt.RemoteEnv`
        Remote environement to interact with.
    """
    env = None
    try:
        additional_args = []
        for k, v in kwargs.items():
            k = k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    additional_args.append(f"--{k}")
                else:
                    additional_args.append(f"--no-{k}")
            else:
                additional_args.extend([f"--{k}", str(v)])

        launcher_args = dict(
            scene=scene,
            script=script,
            num_instances=1,
            named_sockets=["GYM"],
            instance_args=[additional_args],
            background=background,
        )
        with BlenderLauncher(**launcher_args) as bl:
            env = RemoteEnv(bl.launch_info.addresses["GYM"][0])
            yield env
    finally:
        if env:
            env.close()


try:
    import gym
    from contextlib import ExitStack

    class OpenAIRemoteEnv(gym.Env):
        """Base class for remote OpenAI gym compatible environments.

        By inherting from this class you can provide almost all of the
        code necessary to register a remote Blender environment to
        OpenAI gym.

        See the `examples/control/cartpole_gym` for details.

        Params
        ------
        version : str
            Version of this environment.
        """

        metadata = {"render.modes": ["rgb_array", "human"]}

        def __init__(self, version="0.0.1"):
            self.__version__ = version
            self._es = ExitStack()
            self._env = None

        def launch(self, scene, script, background=False, **kwargs):
            """Launch the remote environment.

            Params
            ------
            scene: path, str
                Blender scene file
            script: path, str
                Python script containing environment implementation.
            background: bool
                Whether or not this environment can run in Blender background mode.
            kwargs: dict
                Any keyword arguments passes as command-line arguments
                to the remote environment. See `btt.env.launch_env` for
                details.
            """
            assert not self._env, "Environment already running."
            self._env = self._es.enter_context(
                launch_env(scene=scene, script=script, background=background, **kwargs)
            )

        def step(self, action):
            """Run one timestep of the environment's dynamics. When end of
            episode is reached, you are responsible for calling `reset()`
            to reset this environment's state.

            Accepts an action and returns a tuple (observation, reward, done, info).
            Note, this methods documentation is a 1:1 copy of OpenAI `gym.Env`.

            Params
            ------
            action: object
                An action provided by the agent

            Returns
            -------
            observation: object
                Agent's observation of the current environment
            reward: float
                Amount of reward returned after previous action
            done: bool
                Whether the episode has ended, in which case further step() calls will return undefined results
            info: (dict)
                Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
            """
            assert self._env, "Environment not running."
            obs, reward, done, info = self._env.step(action)
            return obs, reward, done, info

        def reset(self):
            """Resets the state of the environment and returns an initial observation.

            Note, this methods documentation is a 1:1 copy of OpenAI `gym.Env`.

            Returns
            -------
            observation: object
                The initial observation.
            """
            assert self._env, "Environment not running."
            obs, info = self._env.reset()
            return obs

        def seed(self, seed):
            """'Sets the seed for this env's random number generator(s)."""
            raise NotImplementedError()

        def render(self, mode="human"):
            """Renders the environment.

            Note, we consider Blender itself the main vehicle to view
            and manipulate the current environment state. Calling
            this method will usually render a specific camera view
            in Blender, transmit its image and visualize it. This will
            only work, if the remote environment supports such an operation.
            """
            assert self._env, "Environment not running."
            return self._env.render(mode=mode)

        @property
        def env_time(self):
            """Returns the remote environment time."""
            return self._env.env_time

        def close(self):
            """Close the environment."""
            if self._es:
                self._es.close()
                self._es = None
                self._env = None

        def __del__(self):
            self.close()


except ImportError:
    pass
