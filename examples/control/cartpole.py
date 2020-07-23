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
        env.render()
        if done:
            obs = env.reset()
    env.done()

if __name__ == '__main__':
    main()
