import numpy as np
from contextlib import ExitStack
from pathlib import Path

import gym
from gym import spaces

from blendtorch import btt

class CartpoleEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, render_every=10):
        self.__version__ = '0.0.1'
        
        self.action_space = spaces.Box(np.float32(-100), np.float32(100), shape=(1,))
        self.observation_space = spaces.Box(np.float32(-10), np.float32(10), shape=(1,))
        self.viewer = None

        self._es = ExitStack()
        self._env = self._es.enter_context(
            btt.gym.launch_env(
                scene=Path(__file__).parent/'cartpole.blend',
                script=Path(__file__).parent/'cartpole.blend.py',
                render_every=render_every
            )
        )
        
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        return obs
        
    def seed(self, seed):
        pass

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):        
        self._es.close()
        self._es = None