#!/usr/bin/env python3

import argparse
import os
import torch
import numpy as np
import torch.multiprocessing as mp

import my_optim
from wrappers import *
from model import ActorCritic
from test import test
from train import train

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value loss coefficient (default: 40)')
parser.add_argument('--seed', type=int, default=1000,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=32,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='Breakout-v0',
                    help='environment to train on (default: Breakout-v0)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--model-path', default="./model.pt",
                    help='use an optimizer without shared momentum.')
parser.add_argument('--log-dir', default="./logs/",
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    pid = os.getpid()

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name, episode_life=False, frame_stack=True, scale=True, normalize=False, clip_rewards=False)
    shared_model = ActorCritic(4, env.action_space.n)
    shared_model.share_memory()
    print(shared_model)
    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    p = mp.Process(target=test, args=(pid, args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(pid, rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
