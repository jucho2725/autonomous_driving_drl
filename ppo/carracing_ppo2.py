# -*- coding: utf-8 -*-

"""IMPORT PACKAGES"""

'''
This is slightly edited copy of train_cartpole.py
It also includes callback function for monitoring
'''
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL) 

import argparse

import gym
import numpy as np

'''Model selection '''
from stable_baselines import logger

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy

'''Vectorized Env'''
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds

'''Monitoring Learning process'''
import os
import matplotlib.pyplot as plt

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, FrameStack

"""DEFINITION"""

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
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
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward
#
# # Random Agent, before training
# mean_reward_before_train = evaluate(model, num_steps=10000)


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


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

def plot_results_ns(log_folder, title='Learning Curve no smoothing'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
#     y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

"""MAIN FUNCTION"""


def main():
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    print("Making a new model")

        # env_id = 'CarRacing-v0'
    # num_cpu = 4  # Number of processes to use
    # # Create the vectorized environment
    #
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)
    env = gym.make('CarRacing-v0')
    env = MaxAndSkipEnv(env, skip=4)
    # env = FrameStack(env, 4)
    # env = Monitor(env, log_dir, allow_early_resets=True)
    # env = DummyVecEnv([lambda: env])

    env = VecFrameStack(make_atari_env(env_id=env, num_env=, seed=777), 4)
    model = PPO2(policy=CnnLstmPolicy, env=env, n_steps=512, nminibatches=256,
                 lam=0.95, gamma=0.99, noptepochs=8, ent_coef=0.01,
                 learning_rate=5e-4, cliprange= lambda f : f * 0.2, verbose=0, tensorboard_log='graph/')

    '''
    (self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None)Table 1. Hyperparameters of Unity ML-agent 
                 
Hyperparameter// Value //Description //TypicalRange 
1 Gamma 0.995 corresponds to the discount factor for future rewards 0.8 - 0.995 보상값에 민감하도록 값을 높게설정 
2 Lambda 0.95 corresponds to the lambda parameter used when calculating the Generalized Advantage Estimate (GAE) 0.9 - 0.95 보상값을 추정값보다 신뢰하도록 값을 높 게 설정

Buffer Size, Batch Size, Number of Epoch, Number of Layers:: 시뮬레이션의 복잡도를 고려하여 값을 낮게 설정
3 BufferSize 5120 corresponds to how many experiences should be collected before we do any
learning or updating of the model 2048 - 409600
4 BatchSize 512 the number of experiences used for one iteration of a gradient descent update 512 - 5120
5 Number of Epochs 3 the number of passes through the experience buffer during gradient descent 3-10

Learning Rate, Max Steps: 실험을 통해 설정
Time Horizen, Hidden Units: Buffer Size등의 크기를 고려하여 설정
Beta: 엔트로피(Entropy)의 변화를 고려하여 설정
Epsilon: 에이전트의 이동폭을 고려하여 낮게 설정
6 Learning Rate 0.0003 corresponds to the strength of each gradient descent update step 1e-5 - 1e-3
7 Time Horizon 64 corresponds to how many steps of experience to collect per-agent before adding it to the experience buffer 32 - 2048 
8 Max Steps 1.0e5 corresponds to how many steps of the simulation are run during the training process 5e5 - 1e7
9 Beta 1e-3 corresponds to the strength of the entropy regularization, which makes the policy “more random.: This ensures that agents properly explore the action space during training 1e-4 - 1e-2
10 Epsilon 0.1 corresponds to the acceptable threshold of divergence between the old and new policies during gradient descent updating 0.1 - 0.3
11 Number of Layers 1 corresponds to how many hidden layers are present after the observation input, or after the CNN encoding of the visual observation 1 - 3 
12 Units 128 correspond to how many units are in each fully connected layer of the neural network 32 - 512
    
    '''

    # mean_reward_before_train = evaluate(model, env=env, num_steps=10000)
    print("Learning started. It takes some time...")
    model.learn(total_timesteps=300000, callback=callback, tb_log_name='190317')
    # mean_reward = evaluate(model, env=env, num_steps=10000)
    print("Saving model to CarRacing_model.pkl")
    model.save("CarRacing_model_PPO2_5")
    print("Plotting Learning Curve")
    plot_results(log_dir)
    plot_results_ns(log_dir)

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
    trained_model = PPO2.load("CarRacing_model_PPO1_1.pkl")
    trained_model.set_env(env)
    trained_model.learn(300000)
    print("Saving model to CarRacing_model.pkl")
    trained_model.save("CarRacing_model_PPO1_1.5.pkl")
    plot_results(log_dir)

if __name__ == '__main__':
    # Create log dir
    log_dir = "/tmp/190207/"
    os.makedirs(log_dir, exist_ok=True)
    best_mean_reward, n_steps = -np.inf, 0

    main()

    # run()

    # cont_learn()