# -*- coding: utf-8 -*-

"""
@author: Jin Uk,Cho

"""


"""IMPORT PACKAGES"""

import argparse

import gym
import numpy as np

'''Model selection '''
from stable_baselines import logger

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import CnnPolicy,  MlpPolicy

'''Vectorized Env'''
from stable_baselines.common.vec_env import DummyVecEnv

'''Monitoring Learning process'''
import os
import matplotlib.pyplot as plt

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

'''FrameStack, FrameSkip'''
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, FrameStack

"""DEFINITION"""

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps
  :param _locals: (dict)
  :param _globals: (dict)

  from stable-baselines examples
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 100000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


def evaluate(model, env, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes

    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    mean_100ep_reward = round(np.mean(episode_rewards), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    return mean_100ep_reward


'''Plotting'''

def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve', smoothing = True):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot

    from stable-baselines example
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')

    if smoothing:
        y = movingAverage(y, window=50)
    else:
        title = 'Learning Curve no smoothing'

    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

""" Tweak Environment """

from gym import Wrapper
from gym import spaces

low = np.array([-1.0, 0, 0])
high = np.array([1.0, 1.0, 0.2])

class ControlCarRacing(Wrapper):
    def __init__(self, env):
        super(ControlCarRacing, self).__init__(env)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)



"""MAIN FUNCTION"""

def main():
    """
    Train and save the PPO model, for the cartpole problem

    """
    print("Making a new model")


    env = ControlCarRacing(gym.make('CarRacing-v0'))
    env = MaxAndSkipEnv(env, skip=4)
    env = FrameStack(env, 4)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO2(policy=CnnPolicy, env=env, n_steps=128, nminibatches=4,
                 noptepochs=10, learning_rate=3e-4, cliprange= lambda f : f * 0.2, verbose=0, tensorboard_log='graph/')

    print("Learning started. It takes some time...")
    model.learn(total_timesteps=300000, callback=callback, tb_log_name='190317')
    print("Saving model to CarRacing_model.pkl")
    model.save("CarRacing_model_PPO2")
    print("Plotting Learning Curve")
    plot_results(log_dir)
    plot_results(log_dir, smoothing=False)

def run():
    """
    Run a trained model for the pong problem
    """
    env = gym.make('CarRacing-v0')
    env = DummyVecEnv([lambda: env])

    # model = PPO2.load("CarRacing_model_PPO1_"+ str(5) +".pkl", env)
    model = PPO2.load("CarRacing_model_PPO2_5.pkl", env)
    avg_rew = evaluate(model=model, env=env, num_steps=10000)
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)

            episode_rew += rew
        print("Episode reward", episode_rew)

def cont_learn():
    print('Continue learning....')

    env = gym.make('CarRacing-v0')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    trained_model = PPO2.load("CarRacing_model_PPO2.pkl")
    trained_model.set_env(env)
    trained_model.learn(300000)
    print("Saving model to CarRacing_model.pkl")
    trained_model.save("CarRacing_model_PPO2.pkl")
    plot_results(log_dir)




""" Main """

if __name__ == '__main__':
    # Create log dir
    log_dir = "/tmp/190317/"
    os.makedirs(log_dir, exist_ok=True)
    best_mean_reward, n_steps = -np.inf, 0

    main()

    # run()

    # cont_learn()