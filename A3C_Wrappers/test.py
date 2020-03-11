#!/usr/bin/env python3

import time
from collections import deque
import torch
import torch.nn.functional as F
import psutil
import logging

from utils import *
from wrappers import *
from model import ActorCritic


def test(pid, rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name, episode_life=False, frame_stack=True, scale=True, normalize=False, clip_rewards=False)
    # env = gym.wrappers.Monitor(env, './test_model', force=True)
    env.seed(args.seed + rank)
    log = {}
    setup_logger('{}_test_log'.format(args.env_name), args.log_dir)
    log['{}_test_log'.format(args.env_name)] = logging.getLogger('{}_test_log'.format(args.env_name))
    model = ActorCritic(4, env.action_space.n)
    model.eval()
    obs = env.reset()
    reward_sum = 0
    best_score = 0
    done = True
    start_time = time.time()
    step = 0
    # for plot
    reward_list = []
    steps = []
    times = []
    # a quick hack to prevent the agent from stucking
    # In the starting state, the game waits for player(actor) to hit fire (action 1).
    # Game starts with 5 lives;
    # and every time player (actor) fails to return the ball,
    # number of lives goes down and the game waits for fire again.
    # Hence, if you pass constant actions that are not 1, the actual game never starts.
    actions = deque(maxlen=100)
    while True:
        pps = psutil.Process(pid=pid)
        try:
            if pps.status() in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED):
                break
        except psutil.NoSuchProcess:
            break
        state = get_state(obs)
        step += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        with torch.no_grad():
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        obs, reward, done, info = env.step(action[0, 0])
        reward_sum += reward
        actions.append(action[0, 0])
        # all the elements in actions_deque is actions[0] -> done(deadlock)
        if actions.count(actions[0]) == actions.maxlen:
            log['{}_test_log'.format(args.env_name)].info("Deadlock -> Done {} lives remained with action {}".format(info["ale.lives"], actions[0]))
            print("Deadlock -> Done {} lives remained with action {}".format(info["ale.lives"], actions[0]))
            # print(prob)
            done = True
        if done:
            log['{}_test_log'.format(args.env_name)].info("Time {}, already train {} steps, episode reward {} in {} steps".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), counter.value, reward_sum, step))
            print("Time {}, already train {} steps, episode reward {} in {} steps".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), counter.value, reward_sum, step))
            reward_list.append(reward_sum)
            steps.append(counter.value)
            second = int(time.time() - start_time)
            times.append(second/3600)
            show(steps, reward_list, args.env_name, "steps", "score", 'score_steps.png')
            show(times, reward_list, args.env_name, "time(hrs)", "score", 'score_time.png')
            if reward_sum > best_score:
                best_score = reward_sum
                torch.save(model, args.model_path)
            step = 0
            reward_sum = 0
            actions.clear()
            obs = env.reset()
            time.sleep(60)
