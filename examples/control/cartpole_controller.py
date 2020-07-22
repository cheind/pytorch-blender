from contextlib import ExitStack
import time
from blendtorch import btt

def controller(obs):
    c,p,_ = obs
    return (p-c)*25

def main():
    with btt.BlenderLauncher(
        scene=f'cartpole.blend', 
        script=f'cartpole_env.py', 
        num_instances=1, 
        named_sockets=['GYM']
        ) as bl:

        env = btt.gym.RemoteEnv(bl.launch_info.addresses['GYM'][0])
        obs = env.reset()
        N = 0
        t = time.time()
        while True:
            obs, reward, done, info = env.step(controller(obs))
            if done:
                print(done)
                obs = env.reset()                
            N += 1
            # if N % 100 == 0:
            #     print('FPS', N/(time.time()-t))


if __name__ == '__main__':
    main()


# https://github.com/MartinThoma/banana-gym/tree/master/gym_banana
# https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym