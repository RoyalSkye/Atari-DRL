#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from itertools import count
import torch.nn.functional as F
import os
import time
import csv
from collections import deque

# from wrappers import *
from envs import *
from model import ActorCritic


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    path = log_file + logger_name
    fileHandler = logging.FileHandler(path, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def get_state(obs):
    state = np.array(obs)
    # print(state.shape)
    state = state.transpose((2, 0, 1))
    state = state.astype(np.float32)
    state = torch.from_numpy(state)
    # print(state.size())
    return state.unsqueeze(0)


def show(x, y, title, des, xdes, ydes, path):
    plt.plot(x, y, 'b-', label=des)
    plt.xlabel(xdes)
    plt.ylabel(ydes)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close("all")

def save_csv(path, steps, score):
    with open(path, 'a+') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([steps, score])

def play_atari(env_name='Breakout-v0', path='./model.pt', render=False):
    start_time = time.time()
    env = create_atari_env(env_name)
    # env = create_atari_env(env_name, episode_life=False, frame_stack=True, scale=True, normalize=False, clip_rewards=False)
    env = gym.wrappers.Monitor(env, './test_model', force=True)
    # model = ActorCritic(4, env.action_space.n)
    model = torch.load(path)
    print(model)
    env.seed(2020)
    for episode in range(20):
        obs = env.reset()
        total_reward = 0.0
        actions = deque(maxlen=2000)
        done = True
        if done:
            cx = torch.zeros(1, 512)  # 256 for wrappers
            hx = torch.zeros(1, 512)
        else:
            cx = cx.detach()
            hx = hx.detach()

        for t in count():
            state = torch.from_numpy(obs)
            with torch.no_grad():
                value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].numpy()
            if render:
                env.render()
            obs, reward, done, info = env.step(action[0, 0])
            total_reward += reward
            if done:
                print("Finished Episode {} with steps {} reward {}".format(episode + 1, t + 1, total_reward))
                actions.clear()
                break
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                print("Deadlock -> Done {} lives remained with action {}".format(info["ale.lives"], actions[0]))
                print(time.time() - start_time)
                return
            # time.sleep(0.2)
    env.close()
    print(time.time() - start_time)

if __name__ == '__main__':
    play_atari(path='./model/model.pt')
    # save_csv('./steps_average_score.csv', 1, 10)