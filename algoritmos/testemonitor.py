import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('monitoringmeu_computador_pessoal_666.csv')
print(df)
df.plot()

def get_hour(x):
    return x.split(' ')[1]
df['currentTime']=df['currentTime'].apply(get_hour)
plt.show()
