import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the vectorized Gym environment
env = make_vec_env('CartPole-v1', n_envs=1)

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Make the model play
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    env.render()

# Close the environment
env.close()
