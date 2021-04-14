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

SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    # Classification time
fig, axs = plt.subplots(1)

# Making plots of the CNN algorithm
axs.set_xlabel('Image Size')
axs.set_ylabel('Single Classifying Time (s)')
axs.set_title('Single Classifying Time for the CNN')
axs.plot(cnn['Target size'],cnn['Single classifying time (s)'], marker='o')
axs.xaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(0.5))
axs.xaxis.set_minor_locator(AutoMinorLocator(1))
axs.yaxis.set_minor_locator(AutoMinorLocator(1))
axs.grid(which='major', linestyle='--')
axs.grid(which='minor', linestyle=':')




fig, axs = plt.subplots(1)
# Making plots of the RFC algorithm
axs.set_xlabel('Image Size')
axs.set_ylabel('Single Classifying Time (s)')
axs.set_title('Single Classifying Time for the RFC')
axs.plot(rfc[rfc['n Trees']==100]['Target size'], rfc[rfc['n Trees']==100]['Single classifying time (s)'], c='orange', label='n Trees = 100', marker='o')
axs.plot(rfc[rfc['n Trees']==500]['Target size'], rfc[rfc['n Trees']==500]['Single classifying time (s)'], label='n Trees = 500', marker='*', markersize=7)
axs.legend()
axs.xaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(5))
axs.xaxis.set_minor_locator(AutoMinorLocator(0.5))
axs.yaxis.set_minor_locator(AutoMinorLocator(0.5))
axs.grid(which='major', linestyle='--')
axs.grid(which='minor', linestyle=':')


fig, axs = plt.subplots(1)
# Making plots of the KNN algorithm
axs.set_xlabel('Image Size')
axs.set_ylabel('Single Classifying Time (s)')
axs.set_title('Single Classifying Time for the KNN')
axs.plot(knn['Target size'], knn['Single classifying time (s)'], marker='o')
axs.legend()
axs.xaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(0.5))
axs.xaxis.set_minor_locator(AutoMinorLocator(1))
axs.yaxis.set_minor_locator(AutoMinorLocator(1))
axs.grid(which='major', linestyle='--')
axs.grid(which='minor', linestyle=':')



plt.tight_layout()
plt.show()
