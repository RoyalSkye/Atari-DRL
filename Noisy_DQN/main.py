import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import agent 
from config import *
import agent_noisy 
import pdb
import matplotlib.pyplot as plt
import argparse
import json
import time

parser = argparse.ArgumentParser(description='atari-Breakout')
parser.add_argument('--train', action = 'store_true', help = "train the model")
parser.add_argument('--test',  action = 'store_true', help = "test the model")
parser.add_argument('--resume',  action = 'store_true', help = "resume training the model")
parser.add_argument('--noisy',  action = 'store_true', help = "train noisy model")
parser.add_argument('--epsilon',  type = float, default=1.0, help = "epsilon")
parser.add_argument('--best',  type = float, default=0.0, help = "best score so far")
parser.add_argument('--save_path',  type = str, default = 'breakout_dqn4.pt', help = "save path of model")
parser.add_argument('--resume_path',  type = str, default = 'breakout_dqn4.pt', help = "resume path of model")
parser.add_argument('--j_path',  type = str, default = 'j_1.json', help = "path of json training file")
args = parser.parse_args()

def plot_reward(mean_reward):
	labels = [ "mean_reward" ]
	plot_data = dict()
	plot_data["mean_reward"] = mean_reward
	fig, ax = plt.subplots(figsize = (10,8))
	for metric in labels:
		ax.plot( np.arange(len(plot_data[metric])) , np.array(plot_data[metric]), label=metric)
	ax.legend()
	plt.xlabel('no. of episodes')
	fig.savefig(args.save_path + 'reward.png')

def plot_loss(loss):
	labels = [ "loss" ]
	plot_data = dict()
	plot_data["loss"] = loss
	fig, ax = plt.subplots(figsize = (10,8))
	for metric in labels:
		ax.plot( np.arange(len(plot_data[metric])) , np.array(plot_data[metric]), label=metric)
	ax.legend()
	plt.xlabel('no. of episodes')
	fig.savefig(args.save_path + 'loss.png')


env = gym.make('Breakout-v0')

number_lives = find_max_lifes(env)
state_size = env.observation_space.shape
action_size = 4 
rewards, episodes = [], []
ep_rewards, ep_loss = [], []
ep_score = []

evaluation_reward = deque(maxlen=evaluation_reward_length)
evaluation_loss = deque(maxlen=evaluation_reward_length)
frame = 0
memory_size = 0

if args.noisy:
	model_agent = agent_noisy
else:
	model_agent = agent

if args.train:
	if args.resume:
		agent = model_agent.Agent(action_size, load_model=True, epsilon=args.epsilon, model_path="./save_model/" + args.resume_path)
	else:
		agent = model_agent.Agent(action_size)
	best_score = args.best 
	for e in range(EPISODES):
		done = False
		score = 0

		history = np.zeros([5, 84, 84], dtype=np.uint8)
		step = 0
		d = False
		state = env.reset()
		state, _, _, info = env.step(1)
		cur_life = number_lives
		
		# Initial history with state(duplicate 4 times) filling up the first 4 rows
		get_init_state(history, state)

		while not done:
			step += 1
			frame += 1
			#if render_breakout:
			#	env.render()

			# Select and perform an action
			# Need to feed 4 frames into policy_net of agent
			life = info['ale.lives']
			if life == cur_life:
				action = agent.get_action(np.float32(history[:4, :, :]) / 255.)
				next_state, reward, done, info = env.step(action)
			else:
				next_state, reward, done, info = env.step(1)
				cur_life = life

			frame_next_state = get_frame(next_state)
			#assign last row of history with next_state frame
			history[4, :, :] = frame_next_state
			#whether terminal_state has reached (bool)
			terminal_state = check_live(life, info['ale.lives'])

			#r = np.clip(reward, -1, 1)
			r = reward

			# Store the transition in memory 
			# (frame of next_state, action taken to reach next_state, reward at next_state, is next_state terminal?)
			agent.memory.push(deepcopy(frame_next_state), action, r, terminal_state)
			# Start training after random sample generation
			#print(frame)
			if(frame >= train_frame):
				#pdb.set_trace()
				loss = agent.train_policy_net(frame)
				# Update the target network
				if(frame % Update_target_network_frequency)== 0:
					agent.update_target_net()

			score += reward
			#Update history by shifting last 4 frames 1 step forward
			history[:4, :, :] = history[1:, :, :]

			if done:
				evaluation_reward.append(score)
				# every episode, plot the play time
				print("episode:", e, "  score:", score, "  memory length:",
					  len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
					  "evaluation reward:", np.mean(evaluation_reward))
				ep_rewards.append(np.mean(evaluation_reward))
				ep_score.append(score)
				try:
					evaluation_loss.append(loss.item())
				except:
					evaluation_loss.append(0)
				ep_loss.append(np.mean(evaluation_loss))

				#if np.mean(evaluation_reward) > best_score:
				if score > best_score:
					#best_score = np.mean(evaluation_reward)
					best_score = score 
					torch.save(agent.policy_net, "./save_model/" + args.save_path)
				if e % 50 == 0:
					plot_reward(ep_rewards)
					plot_loss(ep_loss)
					training_dict = {'loss': ep_loss, 'reward':ep_rewards, 'epsilon':agent.epsilon, 'best_score': best_score, 'ep_score': ep_score}
					with open("./save_model/" + args.j_path, 'w') as fp:
					    json.dump(training_dict, fp)

# For Test
TEST_EPISODES = 50 
tr = time.time()
if args.test:
	agent = model_agent.Agent(action_size, load_model=True, epsilon=args.epsilon, model_path="./save_model/" + args.save_path)
	agent.policy_net.eval()
	for e in range(TEST_EPISODES):
		done = False
		score = 0

		history = np.zeros([5, 84, 84], dtype=np.uint8)
		step = 0
		#env.seed(2000)
		state = env.reset()
		state, _, _, info = env.step(1)
		cur_life = number_lives

		# Update history with state(duplicate 4 times) filling up the first 4 rows
		get_init_state(history, state)
		
		
		while not done:
			step += 1
			frame += 1
			if render_breakout:
				env.render()
				#print('frame,', frame)

			life = info['ale.lives']
			if life == cur_life:
				action = agent.get_action(np.float32(history[:4, :, :]) / 255.)
				next_state, reward, done, info = env.step(action)
			else:
				next_state, reward, done, info = env.step(1)
				cur_life = life
			
			#print('reward', reward)
			#if reward > 1:
			#	time.sleep(3)
			# Select and perform an action
			# Need to feed 4 frames into policy_net of agent
			frame_next_state = get_frame(next_state)
			#assign last row of history with next_state frame
			history[4, :, :] = frame_next_state
			#whether terminal_state has reached (bool)
			terminal_state = check_live(life, info['ale.lives'])

			life = info['ale.lives']

			score += reward
			#Update history by shifting last 4 frames 1 step forward
			history[:4, :, :] = history[1:, :, :]

			if done:
				evaluation_reward.append(score)
				total_time = (time.time() - tr)/60
				print("episode:", e, "  score:", score, "  memory length:",
					  len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
					  "	evaluation reward:", np.mean(evaluation_reward), 'time:', total_time, 'mins')

	env.close()			
