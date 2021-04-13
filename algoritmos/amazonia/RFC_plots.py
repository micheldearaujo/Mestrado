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


rfc = pd.read_csv(base_dir+'/'+'RFC_Scores_ALL.csv')
fig0, axs = plt.subplots(1,2, sharey=True)
axs[0].set_xlabel('Avg Recall')
axs[0].set_ylabel('Avg Precision')
axs[0].set_title('Precision and Recall As Function of The Trees Number and Image Size for RFC')
axs[0].plot(rfc[rfc['Target size']=='8x8']['Avg Recall'],
         rfc[rfc['Target size']=='8x8']['Avg Precision'],
         marker='o',
         label='8x8')
axs[0].plot(rfc[rfc['Target size']=='16x16']['Avg Recall'],
         rfc[rfc['Target size']=='16x16']['Avg Precision'],
         marker='*',
         label='16x16', markersize=10)
axs[0].plot(rfc[rfc['Target size']=='32x32']['Avg Recall'],
         rfc[rfc['Target size']=='32x32']['Avg Precision'],
         marker='^',
         label='32x32')
axs[0].plot(rfc[rfc['Target size']=='64x64']['Avg Recall'],
         rfc[rfc['Target size']=='64x64']['Avg Precision'],
         marker='+',
         label='64x64', markersize=10)
axs[0].grid(which='major', linestyle='--')
axs[0].grid(which='minor', linestyle=':')
axs[0].legend(title='Target Size')



axs[1].set_xlabel('Avg Recall')
axs[1].set_ylabel('Avg Precision')
axs[1].set_title('Precision and Recall As Function of The Trees Number and Image Sizefor RFC')
axs[1].plot(rfc[rfc['n Trees']==100]['Avg Recall'],
         rfc[rfc['n Trees']==100]['Avg Precision'],
         marker='o',
         label='n Trees = 100')
axs[1].plot(rfc[rfc['n Trees']==500]['Avg Recall'],
         rfc[rfc['n Trees']==500]['Avg Precision'],
         marker='^',
         label='n Trees = 500')
axs[1].grid(which='major', linestyle='--')
axs[1].grid(which='minor', linestyle=':')
axs[1].legend(title='Image size')


plt.xlim(0.875,0.935)
plt.ylim(0.965,0.985)
plt.show()
