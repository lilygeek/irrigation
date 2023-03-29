import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('../../data/hourly.xlsx')

data.columns = ['date','hour','moisture6','moisture12','soilwater','soiltemp','lws','temp','humid','prep','irrigation']

data['hour'] = data['hour'].astype(int)
'''
data = data.drop([0,1])

data = data.drop(['blws','irrigation','stage'],axis=1)

data['datetime'] = data['datetime'].astype(str)
data['moisture6'] = data['moisture6'].astype(float)
data['moisture12'] = data['moisture12'].astype(float)
data['soilwater'] = data['soilwater'].astype(float)
data['soiltemp'] = data['soiltemp'].astype(float)
data['lws'] = data['lws'].astype(float)
data['temp'] = data['temp'].astype(float)
data['humid'] = data['humid'].astype(float)
data['prep'] = data['prep'].astype(float)

data[['date','time']] = data.datetime.str.split(' ',expand=True)

data['hour'] = data.time.str.split(':',expand=True)[0]

data = data.drop(['datetime','time'],axis=1)

data = data.groupby(['date','hour'],as_index=False).agg({'moisture6':'mean','moisture12':'mean','soilwater':'mean','soiltemp':'mean','lws':'mean','temp':'mean','humid':'mean','prep':'sum'})
'''


hdata = pd.read_excel('../../data/hourly_sup.xlsx')

hdata.columns = ['datetime','ref_evp','crop_evp','stage','root_depth','desire']

hdata['datetime'] = hdata['datetime'].astype(str)
hdata[['date','time']] = hdata.datetime.str.split(' ',expand=True)
hdata['hour'] = hdata.time.str.split(':0',expand=True)[0]
hdata = hdata.drop(['datetime','time'],axis=1)
hdata['hour'] = hdata['hour'].astype(int)

new_data =  pd.merge(data, hdata, on=['date','hour'], how='inner')

new_data
new_data.to_excel("../../data/all_hourly.xlsx")


