import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
	debug: bool
	result_folder: str
	model_name: str

	def get_path (self):
		debug_name = "debug_" if self.debug else ""
		full_path = os.path.join(debug_name+self.result_folder, self.model_name)
		return full_path
	
	def create_path (self, path_str):
		# Path(path_str).parent.mkdir(parents=True, exist_ok=True)
		Path(path_str).mkdir(parents=True, exist_ok=True)

	def get_all_dir (self):
		return sorted(next(os.walk(self.get_path()))[1])


	def save_model (self, n_saves, model):
		path_str = os.path.join(self.get_path(), "models_{:03d}".format(n_saves))
		self.create_path(self.get_path())
		print("Saving model at : {}".format(path_str))
		torch.save(model.state_dict(), path_str)

	def load_model (self, n_saves, model):
		path_str = os.path.join(self.get_path(), "models_{:03d}".format(n_saves))
		model.load_state_dict(torch.load(path_str))



class ConvBlock(nn.Module):

	def __init__(self, inp, out, stride, kernel_size):
		super(ConvBlock, self).__init__()
		self.conv1 = nn.Conv1d(inp, out, kernel_size=kernel_size, stride=1, dilation=1, padding="same")
		self.conv2 = nn.Conv1d(out, out, kernel_size=kernel_size, stride=1, dilation=1, padding="same")
		self.bit_conv = nn.Conv1d(out, out, kernel_size=1, stride=stride, bias=False)

		state_dict = self.bit_conv.state_dict()
		state_dict["weight"] = (state_dict["weight"]*0 + np.eye(out).reshape(state_dict["weight"].shape)).detach()
		self.bit_conv.load_state_dict(state_dict)
		for param in self.bit_conv.parameters():
			param.requires_grad = False
	

	def forward(self, x):
		inp = x
		x = F.gelu(self.conv1(x))
		x = F.gelu(self.conv2(x))
		# x = x + inp
		return self.bit_conv(x)

class SimpleNetwork(nn.Module):

	def __init__(self):
		super(SimpleNetwork, self).__init__()
		self.conv1 = ConvBlock(1, 4, 2, 7)
		self.conv2 = ConvBlock(4, 8, 2, 5)
		self.conv3 = ConvBlock(8, 16, 2, 3)
		self.conv4 = ConvBlock(16, 32, 4, 3)
		self.conv5 = ConvBlock(32, 64, 4, 3)
		self.conv6 = ConvBlock(64, 128, 4, 3)
		self.fc1 = nn.Linear(128, 100)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)

		x = torch.transpose(x, 1, 2)
		x = F.sigmoid(self.fc1(x))
		return x

class EasyNetwork(nn.Module):

	def __init__(self):
		super(EasyNetwork, self).__init__()
		self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding="same")
		self.fc1 = nn.Linear(429*5, 256)
		self.fc2 = nn.Linear(256, 88)

	def forward(self, x):
		inp = x
		x = F.gelu(self.conv1(x))
		x = torch.transpose(x, 1, 2).reshape((inp.shape[0], inp.shape[2],-1))
		x = F.gelu(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		
		# x = F.sigmoid(x)
		return x

class FramewiseNetwork(nn.Module):

	def __init__(self):
		super(FramewiseNetwork, self).__init__()
		# self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
		# self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
		# self.conv2bis = nn.Conv2d(32, 32, kernel_size=3, padding="same")
		# # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
		# # self.conv3bis = nn.Conv2d(32, 32, kernel_size=3, padding="same")
		# self.convend = nn.Conv2d(32, 64, kernel_size=3, padding="same")
		# self.fc1 = nn.Linear(429*64, 512)
		# self.fc2 = nn.Linear(512, 88)

		features1 = 16
		self.conv1 = nn.Conv2d(1, features1, kernel_size=3, padding="same")
		self.conv2 = nn.Conv2d(features1, features1, kernel_size=3, padding="same")
		self.conv2bis = nn.Conv2d(features1, features1, kernel_size=3, padding="same")
		# self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
		# self.conv3bis = nn.Conv2d(32, 32, kernel_size=3, padding="same")
		self.convend = nn.Conv2d(features1, features1*2, kernel_size=3, padding="same")
		self.fc1 = nn.Linear(429*features1*2, 128*2)
		# self.lstm = torch.nn.LSTM(input_size=429*features1*2, hidden_size=128, batch_first=True, bidirectional=True)
		self.fc2 = nn.Linear(128*2, 88)

	def forward(self, x):
		inp = x
		# x = F.relu(self.conv1(x))
		# x = F.relu(self.conv2(x))
		# x = F.relu(self.conv3(x))
		# x = torch.transpose(x, 1, 2).reshape((inp.shape[0], inp.shape[2],-1))
		# x = F.relu(self.fc1(x))
		# x = F.sigmoid(self.fc2(x))

		x = F.gelu(self.conv1(x))
		inter = x
		x = F.gelu(self.conv2(x))
		x = F.gelu(self.conv2bis(x))
		x = x + inter
		# inter = x
		# x = F.gelu(self.conv3(x))
		# x = F.gelu(self.conv3bis(x))
		# x = x + inter
		x = F.gelu(self.convend(x))
		x = torch.transpose(x, 1, 2).reshape((inp.shape[0], inp.shape[2],-1))
		x = F.gelu(self.fc1(x))
		# x = self.lstm(x)[0]
		x = F.sigmoid(self.fc2(x))
		
		# x = F.sigmoid(x)
		return x



class FramewiseInpNetwork(nn.Module):

	def __init__(self, use_onsets=False):
		super(FramewiseInpNetwork, self).__init__()

		self.use_onsets = use_onsets
		input_dims = 1 if not self.use_onsets else 2

		if self.use_onsets:
			self.fcy = nn.Linear(88, 429)

		features1 = 16 # 16, 25
		self.conv1 = nn.Conv2d(input_dims, features1, kernel_size=3, padding="same")
		self.conv2 = nn.Conv2d(features1, features1, kernel_size=3, padding="same")
		self.conv2bis = nn.Conv2d(features1, features1, kernel_size=3, padding="same")
		# self.conv3 = nn.Conv2d(features1, features1, kernel_size=3, padding="same")
		# self.conv3bis = nn.Conv2d(features1, features1, kernel_size=3, padding="same")
		self.convend = nn.Conv2d(features1, features1*2, kernel_size=3, padding="same")
		self.fc1 = nn.Linear(429*features1*2, 128*2)
		self.fc2 = nn.Linear(128*2, 88)

	def forward(self, x, y=None):
		inp = x
		if self.use_onsets:
			y = self.fcy(y.detach()).reshape((inp.shape[0], 1, inp.shape[2], -1))
			x = torch.concat([x, y], dim=1)
		x = F.gelu(self.conv1(x))
		inter = x
		x = F.gelu(self.conv2(x))
		x = F.gelu(self.conv2bis(x))
		x = x + inter
		# inter = x
		# x = F.gelu(self.conv3(x))
		# x = F.gelu(self.conv3bis(x))
		# x = x + inter
		x = F.gelu(self.convend(x))
		x = torch.transpose(x, 1, 2).reshape((inp.shape[0], inp.shape[2],-1))
		x = F.gelu(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x

class FullNetwork(nn.Module):

	def __init__(self):
		super(FullNetwork, self).__init__()

		self.onset_network = FramewiseInpNetwork()
		self.frame_network = FramewiseInpNetwork(use_onsets=True)

	def forward(self, x):
		# print(x.shape)
		onsets = self.onset_network(x)
		# print(onsets.shape)
		frames = self.frame_network(x, onsets)
		ret = torch.stack([onsets, frames], dim=1)
		# print(ret.shape)
		return ret