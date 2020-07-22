from contextlib import ExitStack
import time
from blendtorch import btt

def controller(obs):
    c,p,_,_ = obs
    return (p-c)*15

def main():
    with btt.BlenderLauncher(
        scene=f'cartpole.blend', 
        script=f'cartpole_env.py', 
        num_instances=1, 
        named_sockets=['GYM']
        ) as bl:

        env = btt.gym.RemoteEnv(bl.launch_info.addresses['GYM'][0])
        obs, reward, done = env.reset()
        N = 0
        t = time.time()
        while True:
            obs, reward, done = env.step(controller(obs))
            if done:
                obs, reward, done = env.reset()
            N += 1
            if N % 100 == 0:
                print('FPS', N/(time.time()-t))


if __name__ == '__main__':
    main()
