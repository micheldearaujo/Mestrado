import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.optimizers import SGD

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/PycharmProjects/data/amazonia' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Parâmetros do modelo
opt = SGD(lr=0.01, momentum=0.9)
targ_shape = (64,64,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
model_name = 'CNN1_CDA_%s_SGD.h5'%(targ_shape[0])

df16 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(16), index_col='Unnamed: 0')
df32 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(32), index_col='Unnamed: 0')
df64 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(64), index_col='Unnamed: 0')

plt.figure(1)
df16['Avg Precision'].plot(label = 'Avg Precision 16')
df16['Avg Recall'].plot(label ='Avg Recall 16')
df32['Avg Precision'].plot(label = 'Avg Precision 32')
df32['Avg Recall'].plot(label ='Avg Recall 32')
df64['Avg Precision'].plot(label = 'Avg Precision 64')
df64['Avg Recall'].plot(label ='Avg Recall 64')
plt.title('Precision Versus Recall')
plt.xlabel('Threshold')
plt.legend()
plt.show()