"""Cartpole controller that interacts with Blender environment through OpenAI.

Run
    python cartpole.py
from this directory.

Note, the custom environment is registered in `cartpole_gym/__init__.py`. The
file `cartpole_gym/cartpole_env.py` defines a OpenAI compatible Blender environment.
The actual environment logic is implemented in Blender script
`cartpole_gym/cartpole.blend.py` and scene `cartpole_gym/cartpole.blend`.
"""


import gym
import cartpole_gym

KAPPA = 30


def control(obs):
    # Simple P controller defining the error as xpole-xcart
    xcart, xpole, _ = obs
    return (xpole - xcart) * KAPPA


def main():
    # Create the environment. The environment is registered in
    # `cartpole_gym/__init__.py`. Set `real_time=True` when you
    # want to keep the simulation running until the agent response.
    env = gym.make("blendtorch-cartpole-v0", real_time=False)

    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(control(obs))
        env.render()
        if done:
            obs = env.reset()
    env.done()


if __name__ == "__main__":
    main()
