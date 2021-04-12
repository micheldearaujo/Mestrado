import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.optimizers import SGD

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)


# o melhor é 64x64

scores = pd.read_csv(base_dir+'/'+'KNN_Scores_ALL.csv')
sns.set_style('whitegrid')
#fig = sns.FacetGrid(data = scores, col='n_trees', hue='Target_size', palette = 'icefire')
#fig.map(sns.scatterplot, 'Avg Recall', 'Avg Precision', s= 100)
sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size',s =100)
# sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size', palette='icefire', ax=axs[1])
#fig.add_legend()

plt.show()
