import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gym, torch, numpy as np, torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import d3rlpy
import numpy as np
from sklearn.model_selection import train_test_split
from d3rlpy.ope import FQE

from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer

params = [0.3,0.2,0.2,0.3]

data = pd.read_excel('../data/main_hourly_locomos.xlsx')


envs = data[['moisture9','moisture18','moisture24','soilwater','soiltemp','lws','temp','humid','pure_prep']]

irrigation = data['irrigation']

envs = envs.to_numpy()
print(envs)
irrigation = irrigation.to_numpy().reshape(-1,1)
rewards = np.zeros(envs.shape[0]-1)
m_r1 = np.abs(envs[:,3]/2.6-1)
m_r2 = m_r1[1:]
m_r1 = m_r1[0:-1]

m9_v1 = envs[0:-1,0]
m9_v2  = envs[1:,0]
m12_v1 = envs[0:-1,1]
m12_v2  = envs[1:,1]
m24_v1 = envs[0:-1,2]
m24_v2  = envs[1:,2]
water = envs[0:-1,3]
desire_bool = np.zeros(envs.shape[0]-1)
irrigation_bool = irrigation==0
m9v_bool = np.zeros(envs.shape[0]-1)
m12v_bool = np.zeros(envs.shape[0]-1)
m24v_bool = np.zeros(envs.shape[0]-1)
desire_bool[water<1.7] = 1
m9v_bool[(m9_v2-m9_v1)>0] = 1
m12v_bool[(m12_v2-m12_v1)>0] = 1
m12v_bool[(m24_v2-m24_v1)!=0] = 1
rewards = params[0]*desire_bool*(m_r1-m_r2)+(1-desire_bool)*irrigation_bool-params[1]*np.abs(m24_v2-m24_v1)+params[2]*desire_bool*(m9_v2-m9_v1)+params[3]*desire_bool*(m12_v2-m12_v1)


terminals = np.zeros(envs.shape[0]-1)
terminals[23::24]=1


print(envs.shape,irrigation.shape,rewards.shape)

dataset  = d3rlpy.dataset.MDPDataset(
	observations=envs[0:-1,:],
	actions = irrigation[0:-1],
	rewards = rewards,
	terminals = terminals,
	discrete_action = False)

train_episodes,test_episodes = train_test_split(dataset,test_size=0.2,random_state=1)
#td_error = td_error_scorer(ddpg,test_episodes)
#average_value = average_value_estimation_scorer(ddpg,test_episodes)
#initial = initial_state_value_estimation_scorer(ddpg,test_episodes)
#opc = soft_opc_scorer(return_threshold=20)  


model = d3rlpy.algos.BCQ.from_json('../result/model/BCQ_20220729122954/params.json')

model.load_model('../result/model/BCQ_20220729122954/model_2001.pt')

i = 0
for episode in train_episodes:
    print("episode",i)
    j = 0
    for observation in episode.observations:
        print("hour",j)
        action = model.predict([observation])[0]
        value = model.predict_value([observation],[action])[0]
        print(action,value)
        j = j+1
    i = i+1

