import d3rlpy
import numpy as np

params = [0.5,0.2,0.3]

data = pd.read_excel('../../data/all_hourly.xlsx')

envs = data[['moisture6','moisture12','soilwater','soiltemp','lws','temp','humid','ref_evp','crop_evp','prep']]

irrigation = data['irrigation']

envs = envs.to_numpy()
irrigation = envs.to_numpy()
rewards = np.zeros(envs.shape(0)-1)
m_r1 = np.abs(envs[:,2]/1.512-1)
m_r2 = m_r1[1:end,:]
m_r1 = m_r1[0:end-1,:]

m6_v1 = envs[0:end,1]
m6_v2  = envs[1:end-1,1]
m12_v1 = envs[0:end,2]
m12_v2  = envs[1:end-1,2]
water = envs[0:end-1,2]
desire_bool = np.zeros(envs.shape(0)-1)
m6v_bool = np.zeros(envs.shape(0)-1)
desire_bool[water<1.512] = 1
m6v_bool[(m6_v2-m6_v1)>0] = 1
rewards = params[0]*(m_r2-m_r1)+ params[1]*1/np.abs(m12_v2-m12_v1)+params[2]*desire_bool*(m6_v2-m6_v1)

terminals = np.zeros(envs.shape(0)-1)
terminals[167::168] = 1
terminals[end] = 1

dataset  = d3rlpy.dataset.MDPDataset(
	observations=envs[0:end-1,:],
	actions = irrigation,
	rewars = rewards,
	terminals = np.zeros(envs.shape(0)-1),
	episode_terminals = terminals
)