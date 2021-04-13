import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)


cnn = pd.read_csv(base_dir+'/'+'CNN_Scores_ALL.csv')
# #sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size',xlim=(0.2,1.1), ylim=(0.2,1.1))
# sns.jointplot(data=scores, x='Threshold', y='Avg Accuracy', hue='Target_size')
# sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Target_size',xlim=(0.2,1.1), ylim=(0.2,1.1))
# sns.jointplot(data=scores, x='Avg Recall', y='Avg Precision', hue='Threshold')

# Precision Vs Recall Vs Image Size
fig0, axs = plt.subplots(1)
axs.set_xlabel('Avg Recall')
axs.set_ylabel('Avg Precision')
axs.set_title('Precision and Recall As Function of Image Size and Threshold for CNN')
axs.plot(cnn[cnn['Target size']=='8x8']['Avg Recall'],
         cnn[cnn['Target size']=='8x8']['Avg Precision'],
         marker='o',
         label='8x8')
axs.plot(cnn[cnn['Target size']=='16x16']['Avg Recall'],
         cnn[cnn['Target size']=='16x16']['Avg Precision'],
         marker='*',
         label='16x16', markersize=10)
axs.plot(cnn[cnn['Target size']=='32x32']['Avg Recall'],
         cnn[cnn['Target size']=='32x32']['Avg Precision'],
         marker='^',
         label='32x32')
axs.plot(cnn[cnn['Target size']=='64x64']['Avg Recall'],
         cnn[cnn['Target size']=='64x64']['Avg Precision'],
         marker='+',
         label='64x64', markersize=10)
axs.grid(which='major', linestyle='--')
plt.xlim(0,1)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Image size')
plt.show()


# Accuracy versus threshold

fig1, axs = plt.subplots(1)
axs.set_xlabel('Threshold')
axs.set_ylabel('Avg Accuracy')
axs.set_title('Accuracy As Function of Threshold for CNN')
axs.plot(cnn[cnn['Target size']=='8x8']['Threshold'],
         cnn[cnn['Target size']=='8x8']['Avg Accuracy'],
         marker='o',
         label='8x8')
axs.plot(cnn[cnn['Target size']=='16x16']['Threshold'],
         cnn[cnn['Target size']=='16x16']['Avg Accuracy'],
         marker='*',
         label='16x16', markersize=10)
axs.plot(cnn[cnn['Target size']=='32x32']['Threshold'],
         cnn[cnn['Target size']=='32x32']['Avg Accuracy'],
         marker='^',
         label='32x32')
axs.plot(cnn[cnn['Target size']=='64x64']['Threshold'],
         cnn[cnn['Target size']=='64x64']['Avg Accuracy'],
         marker='+',
         label='64x64', markersize=10)
axs.grid(which='major', linestyle='--')
plt.xlim(0,1)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Image size', loc='best')
plt.show()

fig2, axs = plt.subplots(1)
axs.set_xlabel('Avg Recall')
axs.set_ylabel('Avg Precision')
axs.set_title('Precision and Recall As Function of Threshold for CNN')
axs.plot(cnn[cnn['Threshold']==0.1]['Avg Recall'],
         cnn[cnn['Threshold']==0.1]['Avg Precision'],
         marker='o',
         label=0.1)
axs.plot(cnn[cnn['Threshold']==0.2]['Avg Recall'],
         cnn[cnn['Threshold']==0.2]['Avg Precision'],
         marker='*',
         label=0.2, markersize=10)
axs.plot(cnn[cnn['Threshold']==0.3]['Avg Recall'],
         cnn[cnn['Threshold']==0.3]['Avg Precision'],
         marker='^',
         label=0.3)
axs.plot(cnn[cnn['Threshold']==0.4]['Avg Recall'],
         cnn[cnn['Threshold']==0.4]['Avg Precision'],
         marker='+',
         label=0.4, markersize=10)
axs.plot(cnn[cnn['Threshold']==0.5]['Avg Recall'],
         cnn[cnn['Threshold']==0.5]['Avg Precision'],
         marker='o',
         label=0.5)
axs.plot(cnn[cnn['Threshold']==0.6]['Avg Recall'],
         cnn[cnn['Threshold']==0.6]['Avg Precision'],
         marker='*',
         label=0.6, markersize=10)
axs.plot(cnn[cnn['Threshold']==0.7]['Avg Recall'],
         cnn[cnn['Threshold']==0.7]['Avg Precision'],
         marker='^',
         label=0.6)
axs.plot(cnn[cnn['Threshold']==0.8]['Avg Recall'],
         cnn[cnn['Threshold']==0.8]['Avg Precision'],
         marker='+',
         label=0.8, markersize=10)
axs.plot(cnn[cnn['Threshold']==0.9]['Avg Recall'],
         cnn[cnn['Threshold']==0.9]['Avg Precision'],
         marker='o',
         label=0.9)
axs.grid(which='major', linestyle='--')
plt.xlim(0,1)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Threshold')
plt.show()