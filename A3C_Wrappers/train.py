#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torch.optim as optim
import psutil

from utils import *
from wrappers import *
from model import ActorCritic


def train(pid, rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name, episode_life=False, frame_stack=True, scale=True, normalize=False, clip_rewards=False)
    filepath = "./train_model_" + str(rank)
    env = gym.wrappers.Monitor(env, filepath, force=True)
    env.seed(args.seed + rank)

    model = ActorCritic(4, env.action_space.n)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    model.train()
    obs = env.reset()
    state = get_state(obs)
    done = True
    while True:
        # if parent process is killed by "kill -9", child process kill itself
        pps = psutil.Process(pid=pid)
        try:
            if pps.status() in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED):
                break
        except psutil.NoSuchProcess:
            break
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        values = []
        log_probs = []
        rewards = []
        entropies = []
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model((state, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            # sampled from the multinomial probability distribution
            action = prob.multinomial(num_samples=1).detach()  # [[1]]
            log_prob = log_prob.gather(1, action)
            obs, reward, done, _ = env.step(action.numpy())
            # done = (done or episode_length >= params.max_episode_length) # if the episode lasts too long (the agent is stucked), then it is done
            # reward = max(min(reward, 1), -1)  # clamping the reward between -1 and +1
            with lock:
                counter.value += 1
            if done:
                obs = env.reset()
            state = get_state(obs)
            entropies.append(entropy)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                # print("step {} done {}".format(step, done))
                break

        # Gradient = ∇θ′logπ(at|st;θ′)[Rt−V(st;θv) + β∇θ′H(π(st;θ′)]
        # gae-lambda - 1.00
        # entropy-coef - 0.01
        # value-loss-coef - 0.5
        # max-grad-norm - 40
        # gamma - 0.99
        R = torch.zeros(1, 1)  # if done R=[[0]]

        if not done:
            value, _, _ = model((state, (hx, cx)))
            R = value.detach()
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)  # Generalized Advantage Estimation
        for i in reversed(range(len(rewards))):
            # advantege = Q - V
            R = rewards[i] + args.gamma * R  # n-step
            advantage = R - values[i]
            # TODO: Confused
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimation
            td_error = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + td_error
            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()
        # if not work, change pytorch to 1.4.0
        (policy_loss + args.value_loss_coef * value_loss).backward()  # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        # print("from train{} counter = {}".format(rank, counter.value))
