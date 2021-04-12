import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.optimizers import SGD

# Definindo o caminho dos diretorios
base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
#base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)


scores = pd.read_csv(base_dir+'/'+'RFC_Scores_ALL.csv')
#sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size',xlim=(0.2,1.1), ylim=(0.2,1.1))
sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='n_trees', palette='rocket')
sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size',xlim=(0.8,1), ylim=(0.8,1), palette='flare')


plt.show()