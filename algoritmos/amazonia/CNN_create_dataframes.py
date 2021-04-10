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

df8 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(8), index_col='Unnamed: 0')
df16 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(16), index_col='Unnamed: 0')
df32 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(32), index_col='Unnamed: 0')
df64 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(64), index_col='Unnamed: 0')
#df128 = pd.read_csv(base_dir+'/'+'CNN_Scores_%s.csv'%(128), index_col='Unnamed: 0')

plt.figure(1)
df8['Avg Precision'].plot(label = 'Avg Precision 8x8')
df16['Avg Precision'].plot(label = 'Avg Precision 16x16')
df32['Avg Precision'].plot(label = 'Avg Precision 32x32')
df64['Avg Precision'].plot(label = 'Avg Precision 64x64')
#df128['Avg Precision'].plot(label = 'Avg Precision 128x128')
df8['Avg Recall'].plot(label = 'Avg Recall 8x8')
df16['Avg Recall'].plot(label = 'Avg Recall 16x16')
df32['Avg Recall'].plot(label = 'Avg Recall 32x32')
df64['Avg Recall'].plot(label = 'Avg Recall 64x64')
#df128['Avg Recall'].plot(label = 'Avg Recall 128x128')
plt.title('Precision and Recall vs Threshold')
plt.xlabel('Threshold')
plt.legend()

def create_target_size8(x):
    return '8x8'
def create_target_size16(x):
    return '16x16'
def create_target_size32(x):
    return '32x32'
def create_target_size64(x):
    return '64x64'
def create_target_size128(x):
    return '128x128'

df8['Target_size'] = df8['Avg TP'].apply(create_target_size8)
df16['Target_size'] = df16['Avg TP'].apply(create_target_size16)
df32['Target_size'] = df32['Avg TP'].apply(create_target_size32)
df64['Target_size'] = df64['Avg TP'].apply(create_target_size64)
#df128['Target_size'] = df128['Avg TP'].apply(create_target_size128)
df = pd.concat([df8,df16,df32,df64], axis =0)
df = df.reset_index()
df.columns = ['Threshold', 'Avg TP', 'Avg FP', 'Avg TN', 'Avg FN', 'Avg Precision',
       'Avg Recall', 'Avg Accuracy', 'Avg F1_score', 'Target_size']
df.to_csv(base_dir+'/'+'CNN_Scores_ALL.csv')

sns.jointplot(data=df, x='Avg Recall', y='Avg Precision', hue='Target_size',xlim=(0.2,1.1), ylim=(0.2,1.1))
plt.title('Precision Vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()