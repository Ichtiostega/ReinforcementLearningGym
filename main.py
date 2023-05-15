import gym
import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import cv2

class Net(nn.Module):
    def __init__(self, in_num, out_num):
        super(Net, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_num, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, out_num),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(in_num, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1))

    def pick_action(self, observation, collector):
        observation = torch.from_numpy(observation).float()
        probabilities = self.actor(observation)
        distribution = Categorical(probabilities)
        action = distribution.sample()

        collector.states.append(observation)
        collector.actions.append(action)
        collector.action_logarithms.append(
            distribution.log_prob(action))
        return action.item()

    def evaluate(self, observation, action):
        probabilities = self.actor(observation)
        distribution = Categorical(probabilities)
        logarithm_probabilities = distribution.log_prob(action)
        entropy = distribution.entropy()
        Qvalue = self.critic(observation)
        return logarithm_probabilities, torch.squeeze(Qvalue), entropy
