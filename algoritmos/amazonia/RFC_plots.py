import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
from tensorflow.keras.optimizers import SGD

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)


rfc = pd.read_csv(base_dir+'/'+'RFC_Scores_ALL.csv')

#matplotlib.rcParams.update({'font.size': 20})
SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False


markers =['o','*','v','s','X','D','+','>','p']
sizes=['8x8','16x16','32x32','64x64']
linestyles=['dashed','solid','dashdot','dotted']
n_trees = [100,500]
cores=['blue','orange','green','red']

fig0, axs = plt.subplots(1,2, sharey=True)
plt.suptitle('Precision and Recall As Function of The Trees Number and Image Size for RFC')
axs[0].set_xlabel('Avg Recall')
axs[1].set_xlabel('Avg Recall')
axs[0].set_ylabel('Avg Precision')
#axs.set_title('Precision and Recall As Function of The Trees Number and Image Size for RFC')
axs[0].grid(which='major', linestyle='--')
axs[0].grid(which='minor', linestyle=':')
axs[1].grid(which='major', linestyle='--')
axs[1].grid(which='minor', linestyle=':')

for k in range(len(sizes)):
    axs[0].plot(rfc[rfc['Target size']==sizes[k]]['Avg Recall'],
            rfc[rfc['Target size']==sizes[k]]['Avg Precision'],
             marker=markers[k],
                markersize=9,
             label=sizes[k])
for k in range(len(n_trees)):
    axs[1].plot(rfc[rfc['n Trees']==n_trees[k]]['Avg Recall'],
            rfc[rfc['n Trees']==n_trees[k]]['Avg Precision'],
             marker=markers[k],
                markersize=8,
             label=n_trees[k])
axs[0].legend(title='Image Size')
axs[1].legend(title='n Trees')
# axs2=axs.twiny()
# for k in [0,5]:
#     x = (rfc[rfc['Target size']=='8x8']['Avg Recall'][0+k],
#          rfc[rfc['Target size']=='16x16']['Avg Recall'][1+k],
#          rfc[rfc['Target size']=='32x32']['Avg Recall'][2+k],
#          rfc[rfc['Target size']=='64x64']['Avg Recall'][3+k])
#
#     y = (rfc[rfc['Target size']=='8x8']['Avg Precision'][0+k],
#          rfc[rfc['Target size']=='16x16']['Avg Precision'][1+k],
#          rfc[rfc['Target size']=='32x32']['Avg Precision'][2+k],
#          rfc[rfc['Target size']=='64x64']['Avg Precision'][3+k])
#     axs2.scatter(x,y, marker=markers[k],
#                 label=n_trees[k],
#                  color='black',
#                  s=90)

#plt.xlim(0.875,0.935)
#plt.ylim(0.965,0.985)


plt.show()
