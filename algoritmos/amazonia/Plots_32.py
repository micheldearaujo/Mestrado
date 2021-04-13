import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)


# Making plots of the CNN algorithm
cnn_32 = pd.read_csv(base_dir+'/'+'CNN_Scores_32.csv')
plt.scatter(x=cnn_32['Target_size'], y=cnn_32['Single_classification_time(s)'])
plt.title('CNN Classification time in Home Computer')
plt.ylabel('Single Classification Time (s)')
plt.xlabel('Image Size')
plt.show()
