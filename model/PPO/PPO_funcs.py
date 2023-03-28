import pickle
from collections import namedtuple
from itertools import count
import random
from datetime import datetime

import os, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import os

from PPO import PPO

p_v = [10,3]
p_a = [16,6,1]

v_fc = 3.84

v_mad = 3.84*0.9

def inference(agent, state):
    action = agent.select_action(state)

    return action

def reward_cal(agent,action,water):
    if water>=v_fc:
        reward = p_v[0]*(water - v_fc) + p_a[0]*action
    elif v_fc > water and water >= v_mad:
        reward = p_a[1] * action
    else:
        reward = p_v[1] * (v_mad - water) + p_a[2] * action

    reward = -np.squeeze(reward)
    return reward

def buffering(agent, action, state):
    water = state[-1]

    reward = reward_cal(agent,action,water)

    agent.buffer.rewards.append(reward)
    agent.buffer.is_terminals.append(False)
