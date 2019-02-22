import argparse

import gym


from stable_baselines.ddpg import DDPG
from stable_baselines.ppo1 import PPO1
from stable_baselines.ddpg.policies import MlpPolicy
import argparse
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq import DQN
import time

import os
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy


def evaluate(model, env, num_steps=1000):
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
        print('state',_states)
        print('action', action)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)
        print("obs",obs)
        print('rewards',rewards)
        print('dones',dones)
        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
            print(episode_rewards[ep_count])
            ep_count+=1
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

def main(args):
    """
    Run a trained model for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make("LunarLanderContinuous-v2")



    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    model = PPO1.load("ppo_lunar2.pkl", env)
    # model = DDPG.load("ddpg_lunar2.pkl", env)
    # model = DQN.load("dqn_lunar.pkl", env)
    avg_rew = evaluate(model=model, env=env, num_steps=10000)
    print(avg_rew)
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                time.sleep(0.02)
                env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        time.sleep(1)
        print("Episode reward", episode_rew)
        # No render is only used for automatic testing
        if args.no_render:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DDPG on LunarLanderContinuous-v2")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    args = parser.parse_args()
    main(args)
