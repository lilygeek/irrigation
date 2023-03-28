import argparse
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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10


data = pd.read_excel('../../data/10_sackett_weather_for_simulation.xlsx')

data = data[['year','month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]


state_dim = 12
action_dim = 1

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

p=[0.95631999,0.31006727,8.49262761,0.01503441]
'''
p_v = [3,10]
p_a = [8,3,1]
'''
p_v = [10,3]
p_a = [16,6,1]

v_fc = 3.84

v_mad = 3.84*0.9

def simulator(v,at,pt,et):
    v = p[0]*v+p[1]*(at+pt)+p[2]*et+p[3]
    return v


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    has_continuous_action_space = True

    max_single_seq = 30

    action_std = 0.5                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.01        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(1000)  # action_std decay frequency (in num timesteps)
    #update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0 
    #####################################################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "../../result/model/PPO_30_/"
    checkpoint_path = directory + "model_.pth"
    print("loading network from : " + checkpoint_path)

    test_data = data.drop(data[data.year!=2022].index)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    

    total_reward = 0

    while start<len(test_data.index)-1:
        water = 0
        state = []
        for j in range(min(max_single_seq,len(temp_data.index)-start-1)):
            reward = 0
            if j == 0:
                weather = test_data.iloc[[start+j]]
                weather = weather[['month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]
                water = random.uniform(v_mad, v_fc)
                state = weather.values
                state = np.insert(state,0,water)
            if np.sum(np.isnan(state))>0:
                start = start+j+1
                count = count - 1
                break
            action = agent.select_action(state)
            next_water = simulator(water,action,weather['pcpn'].values,weather['rpet'].values)
            print(water,action,next_water)
            weather = temp_data.iloc[[start+j+1]]
            weather = weather[['month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]
            next_state = weather.values
            next_state = np.insert(next_state,0,next_water)
            if next_water>=v_fc:
                reward = p_v[0]*(next_water - v_fc) + p_a[0]*action
            elif v_fc>next_water and next_water>=v_mad:
                reward = p_a[1]*action
            else:
                reward = p_v[1]*(v_mad - next_water)+p_a[2]*action
            reward = -np.squeeze(reward)
            total_reward = total_reward + reward
            water = next_water
            state = next_state
        start = start+max_single_seq
        count = count + 1
        
    ppo_agent.buffer.clear()

    print("============================================================================================")

    avg_test_reward = total_reward / count
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
