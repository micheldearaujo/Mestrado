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

scores = pd.read_csv(base_dir+'/'+'CNN_Scores_ALL.csv')
sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size',xlim=(0.2,1.1), ylim=(0.2,1.1))
plt.show()