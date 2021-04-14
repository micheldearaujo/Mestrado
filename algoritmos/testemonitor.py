import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('monitoringcloud_0.1.csv')
print(df)
df.plot()
plt.show()