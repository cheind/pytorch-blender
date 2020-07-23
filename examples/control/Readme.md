## Classic Control

This directory contains a recreation of OpenAI's `CartPole-v0` environment running in a remote Blender process. It utilizes Blender's physics capabilities to simulate the cartpole. The agent running a PID-controller exerts motor forces (continuous control). The communication is done via **blendtorch**, so it appears as any other native OpenAI environment for the agent.

![](capture.gif)

### Code

```python
import gym
import cartpole_gym

KAPPA = 30

def control(obs):
    xcart, xpole, _ = obs
    return (xpole-xcart)*KAPPA

def main():
    env = gym.make('blendtorch-cartpole-v0')
    
    obs = env.reset()        
    while True:
        obs, reward, done, info = env.step(control(obs))        
        if done:
            obs = env.reset()
    env.done()
```
Code: [cartpole_gym](./cartpole_gym), [cartpole.py](cartpole.py)

### Environment rendering
We consider Blender itself as the main tool to view and (interactively) manipulate the state of the environment. In case you want a separate viewer call `env.render()` during your training loop.

### Running
Make sure you have Blender, **blendtorch** (see main Readme), and OpenAI gym (`pip install gym`) installed. Navigate to `examples/control` and run 
```
python cartpole.py
```



