import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('C:\\Users\\miche\\PycharmProjects\\Mestrado\\algoritmos\\monitoringmeupc_0.1.csv')
df2 = pd.read_csv('C:\\Users\\miche\\PycharmProjects\\Mestrado\\algoritmos\\monitoringmeupc_1.csv')
df3 = pd.read_csv('C:\\Users\\miche\\PycharmProjects\\Mestrado\\algoritmos\\monitoringmeupc_5.csv')


def get_hour(x):
    return x.split(' ')[1]
df1['currentTime']=df1['currentTime'].apply(get_hour)
df2['currentTime']=df2['currentTime'].apply(get_hour)
df3['currentTime']=df3['currentTime'].apply(get_hour)
fig, axs = plt.subplots(1)
axs.plot(df1['currentTime'],
         df1['totalCpuUsage(%)'], label ='dt = 0.1s')
axs.plot(df2['currentTime'],
         df2['totalCpuUsage(%)'], label ='dt = 1s')
axs.plot(df3['currentTime'],
         df3['totalCpuUsage(%)'], label ='dt = 5s')
plt.show()
