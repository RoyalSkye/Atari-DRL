import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

class NoisyLinear(nn.Module):
	def __init__(self, in_features, out_features, std_init=0.4):
		super(NoisyLinear, self).__init__()
		
		self.in_features  = in_features
		self.out_features = out_features
		self.std_init	 = std_init
		
		self.weight_mu	= nn.Parameter(torch.FloatTensor(out_features, in_features))
		self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
		self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
		
		self.bias_mu	= nn.Parameter(torch.FloatTensor(out_features))
		self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
		self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
		
		self.reset_parameters()
		self.reset_noise()
	
	def forward(self, x):
		if self.training: 
			weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
			bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
		#else:
		#	weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
		#	bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
		else:
			weight = self.weight_mu 
			bias   = self.bias_mu   
		
		return F.linear(x, weight, bias)
	
	def reset_parameters(self):
		mu_range = 1 / math.sqrt(self.weight_mu.size(1))
		
		self.weight_mu.data.uniform_(-mu_range, mu_range)
		self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
		
		self.bias_mu.data.uniform_(-mu_range, mu_range)
		self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
	
	def reset_noise(self):
		epsilon_in  = self._scale_noise(self.in_features)
		epsilon_out = self._scale_noise(self.out_features)
		
		self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
		self.bias_epsilon.copy_(self._scale_noise(self.out_features))
	
	def _scale_noise(self, size):
		x = torch.randn(size)
		x = x.sign().mul(x.abs().sqrt())
		return x


class NoisyDQN(nn.Module):
	def __init__(self, action_size):
		super(NoisyDQN, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.noisy1 = NoisyLinear(3136, 512)
		self.noisy2 = NoisyLinear(512, action_size)

	def forward(self, x):
		#pdb.set_trace()
		#for i in range(4):
		#	print (x[0][i].sum())
		#print('\n')
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.noisy1(x.view(x.size(0), -1)))
		return self.noisy2(x)

	def reset_noise(self):
		self.noisy1.reset_noise()
		self.noisy2.reset_noise()

