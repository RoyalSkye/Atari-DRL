#!/usr/bin/env python3

import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import gym
import matplotlib.pyplot as plt

from atari_wrappers import *
# from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global epsilon
    global steps_done
    sample = random.random()
    epsilon -= EPS_DECAY
    if epsilon <= EPS_END:
        epsilon = EPS_END
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # print("steps_done {} eps_threshold: {}".format(steps_done, eps_threshold))
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_num)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    # print(non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch)  # [batch_size, 1]
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # DQN
    # Yt ≡ R + γmaxQ(S, a; θ−) θ−: weights of target_net
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # print(next_state_values.size())  # [batch_size]

    # # DDQN
    # # Yt = Rt+1 + γQ(St+1, argmaxQ(St+1, a; θt); θt')
    # next_action = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
    # # print(next_action.size())
    # next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_action).squeeze(1)
    # # print(next_state_values.size())

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # for para in policy_net.parameters():
    #     if para.grad is not None:
    #         print("para.grad not None1")
    optimizer.step()
    return loss.item()


def get_state(obs):
    state = np.array(obs)
    # print(state.shape)  # 84*84*4
    state = state.transpose((2, 0, 1))
    state = state.astype(np.float32)
    state = torch.from_numpy(state)
    # plt.imshow(state[3])
    # plt.show()
    # time.sleep(2)
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    env = gym.wrappers.Monitor(env, './train_model_scale', force=True)
    start_optimize = False
    loss = 0.0
    score = 0
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for step in count():
            action = select_action(state)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # reset done: if 5 lives over, done = True
            # done = True if info["ale.lives"] == 0 else False
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to(device), next_state, reward.to(device))
            state = next_state
            if steps_done > INITIAL_MEMORY:
                start_optimize = True
                loss += optimize_model()
                if (steps_done - INITIAL_MEMORY) % 1000 == 0:
                    policy_net.loss_list.append(loss / 1000)
                    loss = 0.0
            # optimize_model()
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
            # time.sleep(0.1)
        score += total_reward
        print('Episode {}/{} Step_total {} steps: {} epsilon {} Total reward: {}'.format(episode, n_episodes, steps_done, step + 1, epsilon, total_reward))
        if episode % 10 == 0:
            policy_net.reward_list.append(score/10)
            score = 0
        if episode % 100 == 0 and start_optimize:
            show(policy_net.loss_list, 1000, "loss per 1000 steps", "loss", "steps", "loss_scale.png")
            show(policy_net.reward_list, 10, "score per 10 episodes", "score", "episodes", "score_scale.png")
            torch.save(policy_net, MODEL_PATH)
        # time.sleep(2)
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './test_model', force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            state = next_state
            # print("Episode {} life {} done{}".format(episode + 1, info["ale.lives"], done))
            if done:
                print("Finished Episode {} with steps {} reward {}".format(episode + 1, t + 1, total_reward))
                break
            # time.sleep(0.2)
        # time.sleep(1.0)
    env.close()
    return

def show(y, scale, des, ydes, xdes, path):
    x = [i*scale for i in range(len(y))]
    plt.plot(x, y, 'b-', label=des)
    plt.xlabel(xdes)
    plt.ylabel(ydes)
    plt.legend()
    plt.savefig(path)
    plt.close("all")

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.02
    EXPLORE_STEP = 1000000
    EPS_DECAY = (EPS_START - EPS_END) / EXPLORE_STEP
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 0.0001
    MEMORY_SIZE = 1000000
    INITIAL_MEMORY = 100000
    NUM_EPISODES = 100000
    HEIGHT = 84
    WIDTH = 84
    TEST_EPISODES = 10
    MODEL_PATH = 'dqn_model_scale.pt'

    # create environment
    # See wrappers.py
    env = create_atari_env("Breakout-v0", episode_life=False, frame_stack=True, scale=True, clip_rewards=False)
    epsilon = EPS_START
    steps_done = 0
    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # create networks
    action_num = env.action_space.n
    policy_net = DQN(HEIGHT, WIDTH, action_num).to(device)
    target_net = DQN(HEIGHT, WIDTH, action_num).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    print(policy_net)

    # setup optimizer
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # train model
    train(env, NUM_EPISODES)
    torch.save(policy_net, MODEL_PATH)

    # load GPU model on GPU
    policy_net = torch.load(MODEL_PATH)
    test(env, TEST_EPISODES, policy_net, render=False)

    # load GPU model on CPU
    # model = DQN(HEIGHT, WIDTH, action_num).to(device)
    # Load all tensors onto the CPU device
    # policy_net = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    # policy_net = model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
