# -*- coding: utf-8 -*-
"""
Making pkl file and graph

plotting function is from examples-stable-baselines
Link: https://stable-baselines.readthedocs.io/en/master/guide/examples.html
"""

"""Import policy, RL agent"""

import gym
import numpy as np
#
# from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
# from stable_baselines.ppo1 import PPO1
#
# from stable_baselines.deepq import DQN, MlpPolicy
#
from stable_baselines.ddpg import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy

import os 
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

"""Plotting Function"""
def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results_ns(log_folder, title='Learning Curve'):
    """
    plot the results with no smoothing
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # Truncate x
    print("shape of x before truncate %f" % x.shape) #
    print("shape of y %f" %y.shape)
    x = x[len(x) - len(y):]
    print("shape of x after truncate %f" % x.shape)

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()
    
def plot_results(log_folder, title='Learning Curve'):
    """
    plot the smoothed results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = movingAverage(y, window=10)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

#@title
# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)\

env = gym.make('MountainCarContinuous-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# Add noise for exploration - Parameter noise or Action noise
# check out this paper:https://arxiv.org/abs/1706.01905

# param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.9, desired_action_stddev=0.1)
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(policy=MlpPolicy, gamma=0.995, env=env, action_noise=action_noise, verbose=0)
#
# model = DQN(
#         env=env,
#         policy=MlpPolicy,
#         learning_rate=1e-3,
#         buffer_size=50000,
#         exploration_fraction=0.1,
#         exploration_final_eps=0.02, verbose=1
# )

# model = PPO1(policy=MlpPolicy, gamma=0.995, optim_batchsize=32, env=env, verbose=0)

"""Train the agent and save it"""

# Train the agent
model.learn(total_timesteps=100000)
# Save the agent
model.save("ddpg_lunar_ou")


"""Checking data"""
print(load_results(log_dir))

"""Plotting"""
print("Plotting Learning Curve")

plot_results_ns(log_dir)
# plot_results(log_dir)

