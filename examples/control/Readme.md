## Classic Control

This directory contains a recreation of OpenAI's `CartPole-v0` environment running in a remote Blender process. In contrast top OpenAI's version, this environment leverages Blender's built-in physics engine to simulate the cartpole. The agent operates the cart by applying forces to the cart from a separate process.

All communication is handled by **blendtorch** in the background, so it appears like any other native OpenAI environment for the agent.

<p align="center">
    <img src="etc/capture.gif">
</p>

### Code

```python
import gym
import cartpole_gym

KAPPA = 30

def control(obs):
    xcart, xpole, _ = obs
    return (xpole-xcart)*KAPPA

def main():
    env = gym.make('blendtorch-cartpole-v0', real_time=False)
    
    obs = env.reset()        
    while True:
        obs, reward, done, info = env.step(control(obs))
        if done:
            obs = env.reset()
    env.done()
```
Related code: environment [cartpole_env](./cartpole_env), agent [cartpole.py](cartpole.py)

### Run it
Make sure you have Blender, **blendtorch** (see main [Readme](/Readme.md)), and OpenAI gym (`pip install gym`) installed. Navigate to `examples/control` and run 
```
python cartpole.py
```

### Real-time vs. non real-time
Environments running via **blendtorch** support a real-time execution mode `real_time=True`. When enabled, the simulation continues independent of the time it takes the agent to respond. Enabling this mode will require your agent to take into account any latency that occurs from network transmission and action computation.

### Environment rendering
We consider Blender itself as the main tool to view and (interactively) manipulate the state of the environment. In case you want a separate viewer call `env.render()` during your training loop.





