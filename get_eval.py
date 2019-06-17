"""
Evaluation agent by obtaining mean reward and number of episode during timesteps

evaluating function is from examples-stable-baselines
Link: https://stable-baselines.readthedocs.io/en/master/guide/examples.html
"""

"""## Import policy, RL agent, ..."""

import gym
from gym import Wrapper
from gym import spaces
import numpy as np

# Algorithms
#
# from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
# from stable_baselines.ppo1 import PPO1
#
from stable_baselines.deepq import DQN, MlpPolicy
#
# from stable_baselines.ddpg import DDPG
# from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise
# from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy

import os
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv


"""Discrete carracing """

import gym


NUM_ACTIONS = 4
ALLOWED_ACTIONS = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0.8]
]

class DiscreteCarRacing(Wrapper):
    def __init__(self, env):
        super(DiscreteCarRacing, self).__init__(env)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
    def step(self, action):
        return self.env.step(ALLOWED_ACTIONS[action])



"""Evaluation """
def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    ep_count = 0
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)
        print("steps %f" % i, "rewards %f" % rewards) # Shows reward the agent get at each timestep
        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
            print(episode_rewards[ep_count]) # for accumulated reward in one episode
            ep_count += 1
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

"""Main Function"""
#@title
# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)\

env = gym.make('CarRacing-v0')
env = DiscreteCarRacing(env)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


"""DDPG Algorithm """
# Add some param noise for exploration
# param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
# n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# model1 = DDPG(policy=LnMlpPolicy, gamma=0.995, actor_lr=1e-4, critic_lr=1e-3, env=env, param_noise=param_noise, verbose=1)
# model1 = DDPG(policy=LnMlpPolicy, gamma=0.995, actor_lr=1e-4, critic_lr=1e-3, env=env, action_noise=action_noise, verbose=1)

"""DQN Algorithm """
model2 = DQN(env=env, policy=MlpPolicy, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, verbose=1)

"""PPO Algorithm """
# model3 = PPO1(policy=MlpPolicy, gamma=0.995, optim_batchsize=32, env=env, verbose=0)



"""Get result """
# note : All model has same named policy but from different algorithm. Can not run three models at once.

# ddpg_pr_result = evaluate(model1, num_steps=10000)
# ddpg_ou_result = evaluate(model1.5, num_steps=10000)

dqn_result = evaluate(model2, num_steps=10000)
# ppo_result = evaluate(model3, num_steps=10000)
