#!/usr/bin/env python3

import time
from collections import deque
import torch
import torch.nn.functional as F
import psutil
import logging

from utils import *
# from wrappers import *
from envs import *
from model import ActorCritic


def test(pid, rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    log = {}
    setup_logger('{}_test_log'.format(args.env_name), args.log_dir)
    log['{}_test_log'.format(args.env_name)] = logging.getLogger('{}_test_log'.format(args.env_name))
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.eval()
    obs = env.reset()

    episode_score = 0
    reward_sum = 0
    best_score = 0
    cur_test_best_score = 0
    done = True
    start_time = time.time()
    reward_list = []
    best_score_list = []
    steps = []
    times = []

    # a quick hack to prevent the agent from stucking
    # In the starting state, the game waits for player(actor) to hit fire (action 1).
    # Game starts with 5 lives;
    # and every time player (actor) fails to return the ball,
    # number of lives goes down and the game waits for fire again.
    # Hence, if you pass constant actions that are not 1, the actual game never starts.
    actions = deque(maxlen=2000)
    while True:
        # if parent process is killed by "kill -9", child process kill itself
        pps = psutil.Process(pid=pid)
        try:
            if pps.status() in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED):
                break
        except psutil.NoSuchProcess:
            break
        cur_steps = counter.value
        steps.append(cur_steps)
        model.load_state_dict(shared_model.state_dict())
        for i in range(1, args.test_episode+1):
            for step in count():
                state = torch.from_numpy(obs)
                # Sync with the shared model
                if done:
                    cx = torch.zeros(1, 512)
                    hx = torch.zeros(1, 512)
                else:
                    cx = cx.detach()
                    hx = hx.detach()
                with torch.no_grad():
                    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1, keepdim=True)[1].numpy()
                obs, reward, done, info = env.step(action[0, 0])
                episode_score += reward
                actions.append(action[0, 0])
                # all the elements in actions_deque is actions[0/2/3] without 1(fire) -> done(deadlock)
                if actions.count(actions[0]) == actions.maxlen:
                    # log['{}_test_log'.format(args.env_name)].info("Deadlock -> Done: {} lives remained with action {} and {} steps".format(info["ale.lives"], actions[0], step+1))
                    # print("Deadlock -> Done {} lives remained with action {} and {} step".format(info["ale.lives"], actions[0], step + 1))
                    # print(prob)
                    done = True
                if done:
                    actions.clear()
                    obs = env.reset()
                    break
            reward_sum += episode_score
            if episode_score > cur_test_best_score:
                cur_test_best_score = episode_score
            episode_score = 0

        average_score = reward_sum / args.test_episode
        log['{}_test_log'.format(args.env_name)].info("Time {}, already train {} steps, each {} episodes average_score {}, best_score {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), cur_steps, args.test_episode, average_score, cur_test_best_score))
        # print("Time {}, already train {} steps, each {} episodes average_score {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), counter.value, args.test_episode, average_score))
        reward_list.append(average_score)
        best_score_list.append(cur_test_best_score)
        second = int(time.time() - start_time)
        times.append(second/3600)
        show(steps, reward_list, args.env_name, "average_score in 10 episodes", "steps", "score", 'score_steps.png')
        show(times, reward_list, args.env_name, "average_score in 10 episodes", "time(hrs)", "score", 'score_time.png')
        show(steps, best_score_list, args.env_name, "best_score in 10 episodes", "steps", "score", 'best_score_steps.png')
        show(times, best_score_list, args.env_name, "best_score in 10 episodes", "time(hrs)", "score", 'best_score_time.png')
        if average_score > best_score:
            best_score = average_score
            torch.save(model, './best_model.pt')
        torch.save(model, args.model_path)
        save_csv('./steps_average_score.csv', cur_steps, average_score)
        save_csv('./steps_best_score.csv', cur_steps, cur_test_best_score)
        reward_sum = 0
        cur_test_best_score = 0
        time.sleep(30)
