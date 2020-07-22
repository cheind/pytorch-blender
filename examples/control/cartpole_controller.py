from contextlib import ExitStack
import time
from blendtorch import btt

def controller(obs):
    c,p,_ = obs
    return (p-c)*30

def main():
    with btt.gym.launch_blender_env(
            scene='cartpole.blend', 
            script='cartpole_env.py',
            # Any additional args will be passed per command line to Blender.
            render_every=10) as env:
        
        N, t = 0, time.time()
        obs, info = env.reset()        
        while True:
            obs, reward, done, info = env.step(controller(obs))
            env.render()
            if done:
                obs, info = env.reset()
            N += 1
            if N % 100 == 0:                
                print('FPS', N/(time.time()-t))
                N, t = 0, time.time()
        

if __name__ == '__main__':
    main()


# https://github.com/MartinThoma/banana-gym/tree/master/gym_banana
# https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym