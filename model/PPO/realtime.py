import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import configparser
import requests
import datetime
import httpx
import asyncio
from time import sleep
import sys
from datetime import datetime
import time
from itertools import groupby
from operator import itemgetter
import numpy as np

from PPO import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np

import PPO_funcs

import smtplib
from email.message import EmailMessage

#from PPO import train_online

BaseURL = "https://dongyoun89gl3.iot.ubidots.com"
Token = "BBFF-WaloALfKrN8g9kzn4OKefkqoFcQGWd"
P1 = "60f7fa8f73efc3551fcd3e48"
P2 = "62b9ad9d1d847216e1be5da1"
P3 = "60f9b6a373efc34b71817757"
P4 = "60c17c1c73efc3519d587d7c"
IDs = [P3]
Devices = []

def group_statics(dic_list):
    mean_list = []
    for key,value in groupby(dic_list,key=itemgetter('hour')):
        l = list(map(lambda x:x['value'],value))
        mean_list.append(np.average(l))

    if len(mean_list)>0:
        return np.average(mean_list),np.min(mean_list),np.max(mean_list)
    else:
        return [0,0,0]


class Device:

    def __init__(self, ID, name, irrigation_time,agent):
        self.ID = ID
        self.name = name
        self.url = BaseURL+"/api/v1.6/datasources/"+ self.ID + "/variables?token=" + Token;
        self.time = irrigation_time
        self.agent = agent
        self.temp = []
        self.humid = []
        self.soil1 = []
        self.soil2 = []
        self.par = []
        self.water = []
        self.rpet = []
        self.timesteps = 0
        self.last_state = []
        self.action = 0
        self.last_irrigation = time.time()

    #只适用于weather？ 湿度只看当前的

    def add_weather(self,variable, value, hour, timestamp):
        if variable != [] and variable[-1]['hour'] == self.time and hour!=self.time:
            variable = []
        if variable == []:
            
            variable.append({'value':value,'hour':hour,'timestamp':timestamp})
        elif int(variable[-1]['timestamp']) < int(timestamp):
            variable.append({'value':value,'hour':hour,'timestamp':timestamp})
        else:
            pass
        print(variable)

        return variable

    def add_water(self,value, year, month, hour, timestamp,args,config):

        if self.water != [] and hour!=self.time:

            self.last_irrigation = time.time()
            temp_mean, temp_min, temp_max = group_statics(self.temp)
            humid_mean, humid_min, humid_max = group_statics(self.humid)
            srad,_,_ = group_statics(self.par)
            rpet,_,_ = group_statics(self.rpet)
            water,_,_ = group_statics(self.water)
            print("current water:",water)
            wdsd = 0
            pcpn = 0

            state = np.array([month,rpet,pcpn,temp_max,temp_min,temp_mean,humid_max,humid_min,humid_mean,srad,wdsd,water])

            reward = PPO_funcs.reward_cal(self.agent,self.action,water)
            dt = datetime.fromtimestamp(timestamp/1000)

            print("with state:",self.last_state,",apply ",self.action,", the day before "+str(dt) + " got the irrigation reward of " + str(reward))

            if args.update:
                PPO_funcs.buffering(self.agent,state,action)
                self.timesteps = self.timesteps + 1
                if self.timesteps % 7==0:
                    self.agent.update()
                    #checkpoint_pth = '../../result/model/model_'+str(max_single_seq)+'_'+ str(time.time())[:10] +'.pkl'
                    #agent.save(checkpoint_pth)
                
            self.last_state = state
            
            action = PPO_funcs.inference(self.agent,state)

            self.action = action

            msg = EmailMessage()

            msg.set_content("Current water content is:"+str(water)+", please apply " + str(action) + "inches water to " + self.name + " now.")

            msg['Subject'] = '[Important][Irrigation]'
            msg['From'] = config['DEFAULT']['sender']
            sender = config['DEFAULT']['sender']
            password = config['DEFAULT']['password']

            recipients = config['FULL']['recipients'].split(',')
            msg['To'] = config['FULL']['recipients']

            with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
                s.login(sender,password)
                s.sendmail(sender,recipients,msg.as_string())
                s.quit()

            self.water = []
        elif hour != self.time:
            pass
        elif self.water == [] or int(self.water[-1]['timestamp']) < int(timestamp):
            self.water.append({'value':value,'hour':self.time,'timestamp':timestamp})
       
        else:
            pass
        print("water",self.water)


    async def get_update(self,args,config):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(self.url)

                result = r.json()
                result = result['results']
                water = 0
                water_flag = 0

                for i in result:

                    variableName = i["name"]
                    lastValue = float(i['last_value']['value'])
                    lastTime1 = i['last_activity']
                    lastTime2 = i['last_value']['timestamp']

                    lastTime = max(int(lastTime2),int(lastTime1))


                    dt = datetime.fromtimestamp(lastTime/1000)
            
                
                    if variableName == "temp":
                        print(variableName)
                        self.temp = self.add_weather(self.temp,lastValue,dt.hour,lastTime)
                    elif variableName == "humid":
                        print(variableName)
                        self.humid = self.add_weather(self.humid,lastValue,dt.hour,lastTime)
                    elif variableName == "soil2":
                        water = water + lastValue*0.12
                        water_flag = water_flag+1
                    elif variableName == "soil1":
                        water = water +lastValue*0.12
                        water_flag = water_flag+1
                    elif variableName == "solar radiation":
                        print(variableName)
                        self.par = self.add_weather(self.par,lastValue,dt.hour,lastTime)
                    elif variableName == "rpet":
                        print(variableName)
                        self.rpet = self.add_weather(self.rpet,lastValue,dt.hour,lastTime)
                    else:
                        pass

                if water_flag == 2:
                    self.add_water(water, dt.year, dt.month, dt.hour, lastTime, args,config)
                    water_flag = 0

        except:
            print("Can't get device: "+self.ID)
            #sys.exit(1)

async def main(args,config):
    
    #str(time.time())[:10]
    
    #writer = SummaryWriter('../../result/log/runs/PPO_'+str(max_single_seq)+str(time.time())[:10]+'/')

    base = "potato3"
    gamma = 0.99
    render = False
    seed = 1
    log_interval = 10

    irrigation_time = 10


    num_state = 12
    num_action = 1

    has_continuous_action_space = True
    action_std = 0.5                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.01        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(7)  # action_std decay frequency (in num timesteps)
    #update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 10               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    for ID in IDs:
        name = base# + str(i)

        agent = PPO(num_state,num_action,lr_actor,lr_critic,gamma,K_epochs,eps_clip,has_continuous_action_space,action_std)

        checkpoint_path = args.directory + args.ppo_path + args.model_file
        print("loading network from : " + checkpoint_path + " for potato " + ID)
        agent.load(checkpoint_path)
        Devices.append(Device(ID,name,irrigation_time,agent))

    while True:
        task_list = []

        for device in Devices:
            task = asyncio.create_task(device.get_update(args,config))
            task_list.append(task)

        await asyncio.gather(*task_list)

        sleep(300)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", default=False)
    parser.add_argument("--mailing_config", default='../../configs/mailing.ini')
    parser.add_argument("--directory", default="../../result/model/")
    parser.add_argument("--ppo_path")
    parser.add_argument("--model_file")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.mailing_config)
    asyncio.run(main(args,config))

