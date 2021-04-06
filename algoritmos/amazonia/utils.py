import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
import time
from datetime import timedelta
import joblib

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/PycharmProjects/data/amazonia' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)


# Importando os dados
def load_dataset(dataset_name):
    data = np.load(base_dir+'/'+dataset_name)
    X, y = data['arr_0'], data['arr_1']
    print('Dimensões: ')
    print('X: ',X.shape, '\n y: ', y.shape)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr = Xtr.reshape(Xtr.shape[0], targ_shape[0] * targ_shape[0] * 3)  ## Vamos concatenar os dados das 3 dimensoes em apenas 1 dimensão
    Xte = Xte.reshape(Xte.shape[0], targ_shape[0] * targ_shape[0] * 3)
    # Dividindo o set test em dois, para temos validation+test
    Xval = Xte[:4048, :]
    yval = yte[:4048]
    Xte = Xte[4048:, :]
    yte = yte[4048:]
    # Normalizando os dados entre 0 e 1
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.fit_transform(Xval)
    Xte = scaler.fit_transform(Xte)
    return Xtr, Xte, Xval, ytr, yte, yval

def evaluation(modelo,x, true):
    ypred = modelo.predict(x)
    return f1_score(true, ypred, average='samples')