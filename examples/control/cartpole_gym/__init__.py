from gym.envs.registration import register

register(
    id='blendtorch-cartpole-v0', 
    entry_point='cartpole_gym.envs:CartpoleEnv'
)