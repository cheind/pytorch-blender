from pathlib import Path
import numpy as np
from gym import spaces
from blendtorch import btt

class CartpoleEnv(btt.env.OpenAIRemoteEnv):
    def __init__(self, render_every=10):

        super().__init__(version='0.0.1')
        self.launch(
            scene=Path(__file__).parent/'cartpole.blend',
            script=Path(__file__).parent/'cartpole.blend.py',
            render_every=10,            
        )
        
        self.action_space = spaces.Box(np.float32(-100), np.float32(100), shape=(1,))
        self.observation_space = spaces.Box(np.float32(-10), np.float32(10), shape=(1,))