import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import configparser
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

irrigation_time = 10
data = pd.read_excel('../../data/'+str(irrigation_time)+'_sackett_weather_for_simulation.xlsx')

data = data[['year','month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]


num_state = 12
num_action = 1

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

p=[0.95631999,0.31006727,8.49262761,0.01503441]

def simulator(v,at,pt,et):
    v = p[0]*v+p[1]*(at+pt)+p[2]*et+p[3]
    return v



def main(args,config):
    #agent = PPO()
    config_profile = args.config
    p_v = [eval(i) for i in config[config_profile]['p_v'].split(',')]
    p_a = [eval(i) for i in config[config_profile]['p_a'].split(',')]
    v_fc = eval(config[config_profile]['v_fc'])
    v_mad = eval(config[config_profile]['v_mad'])
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

    random_seed = 0         # set random seed if required (0 = no random seed)
    agent = PPO(num_state,num_action,lr_actor,lr_critic,gamma,K_epochs,eps_clip,has_continuous_action_space,action_std)
    
    writer = SummaryWriter('../../result/log/runs/PPO_'+str(irrigation_time)+'_'+str(max_single_seq)+'_'+config_profile+'/')

    #data = pd.read_excel('../data/10_sackett_weather_for_simulation.xlsx')

    train_data = data.drop(data[data.year==2022].index)
    test_data = data.drop(data[data.year!=2022].index)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")
    
    path = '../../result/model/PPO_'+str(max_single_seq)+'_'+str(irrigation_time)+'_'+config_profile+'/'
    
    os.makedirs(path)
    for i_epoch in range(1000):
        count = 0
        total_reward = 0
        total_water = 0
        for i in range(2012,2022):
            start = 0
            temp_data = train_data[train_data.year == i]
            while start<len(temp_data.index)-1:
                water = 0
                state = []
                for j in range(min(max_single_seq,len(temp_data.index)-start-1)):
                    reward = 0
                    if j == 0:
                        weather = temp_data.iloc[[start+j]]
                        weather = weather[['month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]
                        water = random.uniform(v_mad, v_fc)
                        state = np.array(weather.values)
                        state = np.insert(state,0,water)
                    if np.sum(np.isnan(state))>0:
                        start = start+j+1
                        count = count - 1
                        break
                    #action, action_prob = agent.select_action(state)
                     # select action with policy
                    action = agent.select_action(state)

                    next_water = simulator(water,action,weather['pcpn'].values,weather['rpet'].values)
                    weather = temp_data.iloc[[start+j+1]]
                    weather = weather[['month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]
                    next_state = np.array(weather.values)
                    next_state = np.insert(next_state,0,next_water)
                    if next_water>=v_fc:
                        reward = p_v[0]*(next_water - v_fc) + p_a[0]*action
                    elif v_fc>next_water and next_water>=v_mad:
                        reward = p_a[1]*action
                    else:
                        reward = p_v[1]*(v_mad - next_water)+p_a[2]*action
                    reward = -np.squeeze(reward);
                    #print(water,action,next_water,reward)

                    total_reward = total_reward + reward
                    total_water = total_water + next_water
                    #trans = Transition(state, action, action_prob, reward, next_state)
                    #agent.store_transition(trans)
                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(j == min(max_single_seq,len(temp_data.index)-start-1))
                    
                    #print(len(agent.buffer.states),len(agent.buffer.rewards))
                    water = next_water
                    state = next_state
                start = start+max_single_seq 
                count = count + 1
                agent.update()
        print("training Epoch : {} \t\t Average Reward : {} \t\t Average Water : {}".format(i_epoch, total_reward/count, total_water/count))
        writer.add_scalar('reward/train', total_reward/count, global_step=i_epoch)
        writer.add_scalar('water/train', total_water/count, global_step=i_epoch)
        if (i_epoch+1)%3 == 0:
            agent.decay_action_std(action_std_decay_rate, min_action_std)

        if i_epoch%10==0:
            checkpoint_pth = path+'model_'+ str(i_epoch) +'.pkl'
            agent.save(checkpoint_pth)
        start = 0
        total_reward = 0
        total_water = 0
        count = 0
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
                #print(water,action,next_water)
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
                total_water = total_water + next_water
                water = next_water
                state = next_state
            start = start+max_single_seq
            count = count + 1
        writer.add_scalar('reward/test', total_reward/count, global_step=i_epoch)
        writer.add_scalar('water/test', total_water/count, global_step=i_epoch)
        print("testing Epoch : {} \t\t Average Reward : {} \t\t Average Water : {}".format(i_epoch, total_reward/count,  total_water/count))
        agent.buffer.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default='../../configs/train_simulation.ini')
    parser.add_argument("--config", default='potato3')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    main(args,config)
    print("end")
