import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import pickle
from collections import namedtuple
from itertools import count
import random

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


# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10


irrigation_time = 10
data = pd.read_excel('../data/'+str(irrigation_time)+'_weather_for_simulation.xlsx')

data = data[['year','month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]


num_state = 12
num_action = 1

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

#p=[-0.69645153,2.95406154,-13.74428497,2.50611139]
p=[0.30354848,2.95406195,-13.74427775,2.50611132]
p_v = [3,10]
p_a = [8,3,1]
v_fc = 3.84

v_mad = 3.84*0.9

def simulator(v,at,pt,et):
    v = p[0]*v+p[1]*(at+pt)+p[2]*et+p[3]
    return v
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 256)
        self.action_head = nn.Linear(256, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 80
    buffer_capacity = 8000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        path = '../result/log/runs/'+str(time.time())[:10]+'/'
        os.makedirs(path)
        self.writer = SummaryWriter(path)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.001)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 0.003)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)

        #c = Categorical(action_prob)
        #action = c.sample()
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../result/model/actor_net' + str(time.time())[:10] +'.pkl')
        torch.save(self.critic_net.state_dict(), '../result/model/critic_net' + str(time.time())[:10] +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('train_loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('train_loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def main():
    agent = PPO()
    #data = pd.read_excel('../data/10_weather_for_simulation.xlsx')

    train_data = data.drop(data[data.year==2022].index)
    test_data = data.drop(data[data.year!=2022].index)


    for i_epoch in range(1000):
        total_reward = 0
        count = 0
        for i in range(2012,2022):
            start = 0
            temp_data = train_data[train_data.year == i]
            while start<len(temp_data.index)-1:
                water = 0
                state = []
                for j in range(min(30,len(temp_data.index)-start-1)):
                    reward = 0
                    if j == 0:
                        weather = temp_data.iloc[[start+j]]
                        weather = weather[['month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]
                        water = random.uniform(0.5*v_fc, v_fc)
                        state = np.array(weather.values)
                        state = np.insert(state,0,water)
                    if np.sum(np.isnan(state))>0:
                        start = start+j+1
                        break
                    action, action_prob = agent.select_action(state)

                    next_water = simulator(water,action,weather.iloc[0]['pcpn'],weather.iloc[0]['rpet'])
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
                    total_reward = total_reward + reward
                    trans = Transition(state, action, action_prob, reward, next_state)
                    agent.store_transition(trans)
                    water = next_water
                    state = next_state
                start = start+30
                count = count + 1
        agent.update(i_epoch)

        agent.writer.add_scalar('train_avg_total_reward', total_reward/count, global_step=i_epoch)

        start = 0
        total_reward = 0
        count = 0
        while start<len(test_data.index)-1:
            water = 0
            state = []
            for j in range(min(30,len(temp_data.index)-start-1)):
                reward = 0
                if j == 0:
                    weather = test_data.iloc[[start+j]]
                    weather = weather[['month','rpet','pcpn','max_temp','min_temp','mean_temp','max_hum','min_hum','mean_hum','srad','wspd']]
                    water = random.uniform(0.5*v_fc, v_fc)
                    state = weather.values
                    state = np.insert(state,0,water)
                if np.sum(np.isnan(state))>0:
                    start = start+j+1
                    break
                action, action_prob = agent.select_action(state)
                next_water = simulator(water,action,weather.iloc[0]['pcpn'],weather.iloc[0]['rpet'])
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
                total_reward = total_reward + reward
                water = next_water
                state = next_state
            start = start+30
            count = count + 1
            agent.writer.add_scalar('test_avg_total_reward', total_reward/count, global_step=i_epoch)
            if i_epoch%10==0:
                agent.save_param()

if __name__ == '__main__':
    main()
    print("end")
