import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import *
from utils import *
from config import *
import pdb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
	def __init__(self, action_size, epsilon=1.0, load_model=False, model_path=None):
		self.load_model = load_model 

		self.action_size = action_size

		# These are hyper parameters for the DQN
		self.discount_factor = 0.99
		self.epsilon = epsilon 
		self.epsilon_min = 0.01
		self.explore_step = 1000000
		self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
		self.train_start = 100000
		self.update_target = 1000

		# Generate the memory
		self.memory = ReplayMemory()

		# Create the policy net and the target net
		self.policy_net = DQN(action_size)
		self.policy_net.to(device)
		self.target_net = DQN(action_size)
		self.target_net.to(device)

		self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

		# initialize target net
		self.update_target_net()

		if self.load_model:
			self.policy_net = torch.load(model_path, map_location=device)
			self.target_net = torch.load(model_path, map_location=device)

		#self.target_net.eval()
	# after some time interval update the target net to be same with policy net
	def update_target_net(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())

	"""Get action using policy net using epsilon-greedy policy"""
	def get_action(self, state):
		if np.random.rand() <= self.epsilon: 
			# Choose a random action
			a = random.randrange(self.action_size) # explore
		else:
			### CODE ####
			state = torch.from_numpy(state).unsqueeze(0).to(device)
			with torch.no_grad():
				a = self.policy_net(state).argmax(dim=1).detach().cpu().numpy()[0] # exploit
			# 
		return a

	# pick samples randomly from replay memory (with batch_size)
	def train_policy_net(self, frame):
		if self.epsilon > self.epsilon_min:
			self.epsilon -= self.epsilon_decay

		mini_batch = self.memory.sample_mini_batch(frame)
		mini_batch = np.array(mini_batch).transpose()

		history = np.stack(mini_batch[0], axis=0) #shape: (batch_size,5,84,84)
		states = np.float32(history[:, :4, :, :]) / 255. #current state consists of frame(0 to 3)
		actions = list(mini_batch[1])
		rewards = list(mini_batch[2])
		next_states = np.float32(history[:, 1:, :, :]) / 255. #next state consists of frame(1 to 4)
		dones = mini_batch[3] # checks if the game is over
		
		current_q_values = QValues.get_current(self.policy_net, states, actions)
		#use target_net to predict maxQ(S_t, A)
		next_q_values = QValues.get_next(self.target_net, next_states, dones)
		rewards = torch.from_numpy(np.float32(np.array(rewards))).to(device)
		target_q_values = (next_q_values * self.discount_factor) + rewards

		#loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
		loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		return loss

class QValues():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	@staticmethod
	def get_current(policy_net, states, actions):
		states = torch.from_numpy(states).to(device)
		actions = torch.from_numpy(np.array(actions)).to(device)
		return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
	
	@staticmethod		
	# find q_values of states that are NOT terminal states
	# q_values of terminal states are kept at 0
	def get_next(target_net, next_states, dones):				
		next_states = torch.from_numpy(next_states).to(device)
		dones = torch.from_numpy(dones.astype(bool)).to(device)

		non_final_state_locations = (dones == False)
		non_final_states = next_states[non_final_state_locations]
		batch_size = next_states.shape[0]
		values = torch.zeros(batch_size).to(QValues.device)
		values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
		return values
