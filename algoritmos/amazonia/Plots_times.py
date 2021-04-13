import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

cnn = pd.read_csv(base_dir+'/'+'CNN_Scores_03.csv')
knn = pd.read_csv(base_dir+'/'+'KNN_Scores_ALL.csv')
rfc = pd.read_csv(base_dir+'/'+'RFC_Scores_ALL.csv')
cnn.dropna(inplace=True)
knn.dropna(inplace=True)
rfc.dropna(inplace=True)


    # Classification time
fig, axs = plt.subplots(1,3)
fig.suptitle('Single Classifying Time at Home Computer')

# Making plots of the CNN algorithm
axs[0].set_xlabel('Image Size')
axs[0].set_ylabel('Single Classifying Time (s)')
axs[0].set_title('CNN')
axs[0].plot(cnn['Target size'],cnn['Single classifying time (s)'], marker='o')
axs[0].xaxis.set_major_locator(MultipleLocator(1))
axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
axs[0].xaxis.set_minor_locator(AutoMinorLocator(1))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(1))
axs[0].grid(which='major', linestyle='--')
axs[0].grid(which='minor', linestyle=':')
axs[0].legend()

# Making plots of the KNN algorithm
axs[1].set_xlabel('Image Size')
axs[1].set_ylabel('Single Classifying Time (s)')
axs[1].set_title('KNN')
axs[1].plot(knn['Target size'], knn['Single classifying time (s)'], marker='o')
axs[1].legend()
axs[1].xaxis.set_major_locator(MultipleLocator(1))
axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
axs[1].xaxis.set_minor_locator(AutoMinorLocator(1))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(1))
axs[1].grid(which='major', linestyle='--')
axs[1].grid(which='minor', linestyle=':')


# Making plots of the RFC algorithm
axs[2].set_xlabel('Image Size')
axs[2].set_ylabel('Single Classifying Time (s)')
axs[2].set_title('RFC')
axs[2].plot(rfc[rfc['n Trees']==100]['Target size'], rfc[rfc['n Trees']==100]['Single classifying time (s)'], c='orange', label='n Trees = 100', marker='o')
axs[2].plot(rfc[rfc['n Trees']==500]['Target size'], rfc[rfc['n Trees']==500]['Single classifying time (s)'], label='n Trees = 500', marker='*', markersize=7)
axs[2].legend()
axs[2].xaxis.set_major_locator(MultipleLocator(1))
axs[2].yaxis.set_major_locator(MultipleLocator(5))
axs[2].xaxis.set_minor_locator(AutoMinorLocator(0.5))
axs[2].yaxis.set_minor_locator(AutoMinorLocator(0.5))
axs[2].grid(which='major', linestyle='--')
axs[2].grid(which='minor', linestyle=':')

plt.tight_layout()
plt.show()
