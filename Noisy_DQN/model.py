import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class DQN(nn.Module):
	def __init__(self, action_size):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.fc = nn.Linear(3136, 512)
		self.head = nn.Linear(512, action_size)

	def forward(self, x):
		#pdb.set_trace()
		#for i in range(4):
		#	print (x[0][i].sum())
		#print('\n')
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.fc(x.view(x.size(0), -1)))
		return self.head(x)
