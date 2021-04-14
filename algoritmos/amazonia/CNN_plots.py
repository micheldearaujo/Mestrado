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

markers =['o','*','v','s','X','D','+','>','p']
sizes=['8x8','16x16','32x32','64x64']
linestyles=['dashed','solid','dashdot','dotted']
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
cores=['blue','orange','green','red']


# Precision Vs Recall Vs Image Size
fig0, axs = plt.subplots()
axs.set_xlabel('Avg Recall')
axs.set_ylabel('Avg Precision')
axs.set_title('Precision and Recall As Function of Image Size and Threshold for CNN')


# Threshold X Image Size
for k in range(len(sizes)):
    axs.plot(cnn[cnn['Target size']==sizes[k]]['Avg Recall'],
            cnn[cnn['Target size']==sizes[k]]['Avg Precision'],
             ls=linestyles[k],
             label=sizes[k])

axs2=axs.twinx()
for k in range(0,9):
    x = (cnn[cnn['Target size']=='8x8']['Avg Recall'][0+k],
         cnn[cnn['Target size']=='16x16']['Avg Recall'][9+k],
         cnn[cnn['Target size']=='32x32']['Avg Recall'][18+k],
         cnn[cnn['Target size']=='64x64']['Avg Recall'][27+k])

    y = (cnn[cnn['Target size']=='8x8']['Avg Precision'][0+k],
         cnn[cnn['Target size']=='16x16']['Avg Precision'][9+k],
         cnn[cnn['Target size']=='32x32']['Avg Precision'][18+k],
         cnn[cnn['Target size']=='64x64']['Avg Precision'][27+k])
    axs2.scatter(x,y, marker=markers[k],
                label=thresholds[k],
                c='black')

axs.grid(which='major', linestyle='--')
plt.xlim(0.29,1)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Image size', loc=6)
axs2.grid(which='major', linestyle='--')
axs2.grid(which='minor', linestyle=':')
axs2.legend(title='Threshold', loc=3)
plt.show()

# Accuracy versus threshold

fig1, axs = plt.subplots(1)
axs.set_xlabel('Threshold')
axs.set_ylabel('Avg Accuracy')
axs.set_title('Accuracy As Function of Threshold for CNN')

for j in range(len(sizes)):
    axs.plot(cnn[cnn['Target size']==sizes[j]]['Threshold'],
             cnn[cnn['Target size']==sizes[j]]['Avg Accuracy'],
             ls=linestyles[j],
             marker=markers[j],
             label=sizes[j])

axs.grid(which='major', linestyle='--')
plt.xlim(0.09,.91)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Image size', loc=4)
plt.show()
# -----------------------------




## -- Este terceiro grafico nao é mais necessário -- #

# fig2, axs = plt.subplots(1)
# axs.set_xlabel('Avg Recall')
# axs.set_ylabel('Avg Precision')
# axs.set_title('Precision and Recall As Function of Threshold for CNN')
#
# for h in range(len(thresholds)):
#     axs.plot(cnn[cnn['Threshold']==thresholds[h]]['Avg Recall'],
#              cnn[cnn['Threshold']==thresholds[h]]['Avg Precision'],
#              marker=markers[h],
#              label=thresholds[h])
#
# axs.grid(which='major', linestyle='--')
# plt.xlim(0.29,1)
# axs.grid(which='minor', linestyle=':')
# axs.legend(title='Threshold', loc=3)
# plt.show()