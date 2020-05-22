# basic env
import os
import copy
import random
import datetime
import traceback

# plot
import matplotlib
import matplotlib.pyplot as plt

# gym env
import gym
from gym import logger as gymlogger
gymlogger.set_level(40) # error only
import pybulletgym  # register PyBullet enviroments with open ai gym

# tested: numpy 1.18.3
import numpy as np

# tested: PyTorch 1.5.0
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda

# for replay buffer
from collections import namedtuple


# common function
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def draw_training_rewards(title, reward, smooth_len=11, figsize=None):
    def smooth(x, window_len=11):
        if window_len < 3:
            return x
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        w = np.ones(window_len, 'd') / window_len
        y = np.convolve(w, s, mode='valid')
        return y
        
    if figsize:
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.plot(smooth(reward, smooth_len))
    plt.ylabel('Reward')
    plt.xlabel('episode')
    plt.show()
    
    
def load_state(env_name, agent):
    # load previous model
    path = f"model/{agent.name}/{agent.name}_{env_name}.pth"
    if os.path.exists(path):
        checkpoint = torch.load(path)
        agent.load(checkpoint)
        epoch = checkpoint['epoch']
        rewards = checkpoint['rewards']
        if len(rewards):
            max_reward = max(rewards)
        else:
            max_reward = -np.Inf
    else:
        rewards = []
        max_reward = -np.Inf
        epoch = 0
    return epoch, rewards, max_reward


def save_state(env_name, agent, postfix, epoch, rewards):
    path = f"model/{agent.name}/{agent.name}_{env_name}{postfix}.pth"
    model = agent.save()
    model['epoch'] = epoch
    model['rewards'] = rewards
    torch.save(model, path)
    print(f"Save model and {len(rewards)} rewards to {path}.")

    
def evaluate_agent(env_name, agent, config, seed=0):
    set_seeds(seed)
    
    # init env
    env = gym.make(env_name)
    epoch, rewards, max_reward = load_state(env_name, agent)
    print(f"Test {agent.name} at epoch {epoch}, max_reward: {max_reward}")
    
    test_rewards = []
    for t in range(config['test_max_epoch']):
        try:         
            # test
            test_reward = 0
            obs = env.reset()
            for i in range(config.get('test_max_step', 1000)):
                s = obs.ravel()
                a = agent.best(s)
                obs, r, done, _ = env.step(a)
                test_reward += r
                if done:
                    break
            test_rewards.append(test_reward)
            print(f"Test {t + 1} | Step: {i + 1} | Test reward: {test_reward:.1f}")
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            break
    # close env
    env.close()
    print(f"Mean reward: {np.mean(test_rewards):.1f}.")
    draw_training_rewards(f"{agent.name} - {env_name}", rewards, config['plt_smooth'])
