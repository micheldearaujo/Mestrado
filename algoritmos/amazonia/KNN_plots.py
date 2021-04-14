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

knn = pd.read_csv(base_dir+'/'+'KNN_Scores_ALL.csv')
fig0, axs = plt.subplots(1)
axs.set_xlabel('Avg Recall')
axs.set_ylabel('Avg Precision')
axs.set_title('Precision and Recall As Function of Image Size and Threshold for KNN')
axs.plot(knn['Avg Recall'],
         knn['Avg Precision'],
         c='black')

axs.scatter(knn[knn['Target size']=='8x8']['Avg Recall'],
         knn[knn['Target size']=='8x8']['Avg Precision'],
         marker='D',
         label='16x16', s=80)
axs.scatter(knn[knn['Target size']=='16x16']['Avg Recall'],
         knn[knn['Target size']=='16x16']['Avg Precision'],
         marker='X',
         label='16x16', s=80)
axs.scatter(knn[knn['Target size']=='32x32']['Avg Recall'],
         knn[knn['Target size']=='32x32']['Avg Precision'],
         marker='>',
         label='32x32', s=80)
axs.scatter(knn[knn['Target size']=='64x64']['Avg Recall'],
         knn[knn['Target size']=='64x64']['Avg Precision'],
         marker='v',
         label='64x64', s=80)
axs.grid(which='major', linestyle='--')
plt.xlim(0.535,0.5525)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Image size')
plt.show()