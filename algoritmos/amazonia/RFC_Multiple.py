# K Means Clustering é um algoritmo não supervisionado de machine learning que
# agrupa grupo de dados similares.
# Vamos rodar o dataset da amazonia pra ver o que acontece

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import joblib

# Definindo o caminho dos diretorios
# base_dir = '/home/michel/PycharmProjects/data/amazonia' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)



# Importando os dados
def load_dataset(dataset_name):
    data = np.load(base_dir + '/' + dataset_name)
    X, y = data['arr_0'], data['arr_1']
    print('Dimensões: ')
    print('X: ', X.shape, '\n y: ', y.shape)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    Xtr = Xtr.reshape(Xtr.shape[0], targ_shape[0] * targ_shape[0] * 3)  ## Vamos concatenar os dados das 3 dimensoes em apenas 1 dimensão
    Xte = Xte.reshape(Xte.shape[0], targ_shape[0] * targ_shape[0] * 3)
    # Dividindo o set test em dois, para temos validation+test
    Xval = Xte[:4048, :]
    yval = yte[:4048]
    # Normalizando os dados entre 0 e 1
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.fit_transform(Xval)
    Xte = scaler.fit_transform(Xte)
    return Xtr, Xval, ytr, yval


def evaluation(x, true):
    ypred = rfc.predict(x)
    return f1_score(true, ypred, average='samples')


def create_train_model(estimators):
    start_time = time.monotonic()
    # Creating and fitting the model
    rfc = RandomForestClassifier(n_estimators=estimators, verbose=1, oob_score=True)
    rfc.fit(Xtr, ytr)

    # Validation set
    prev_val = evaluation(Xval, yval)
    score_val = rfc.score(Xval, yval)

    # Test set
    # prev_te = evaluation(Xte, yte)
    # score_te = rfc.score(Xte, yte)

    end_time = time.monotonic()
    tempo = timedelta(seconds=end_time - start_time)
    print('Tempo do treinamento: ')
    print('\n')
    print(timedelta(seconds=end_time - start_time))
    print('Amazon Dataset: ', targ_shape)
    print('F1_score_validation: ', prev_val)
    print('Score_validation: ', score_val)

    # Salvando o modelo
    filename = 'RFC_%s_%s.sav' % (targ_shape[0], estimators)
    joblib.dump(rfc, base_dir + '/' + filename)

    file = open(base_dir + '/' + 'RFC_Training.txt', 'a')
    file.write('Image Size: %s_%s\n' % (targ_shape[0], estimators))
    file.write('Training time: %s\n' % tempo)
    file.write('F1_Score_Validation: %s\n' % prev_val)
    file.write('Score_Validation: %s\n' % score_val)
    file.write('----------------------------------------------------\n')
    file.close()

# Calling the functions multiple times to train multiple models while we sleep

# Definindo os parametros
targ_shape = (32, 32)
dataset_name = 'amazon_data_%s.npz' % (targ_shape[0])
estimators = 500

# Loading the dataset
Xtr, Xval, ytr, yval = load_dataset(dataset_name)
create_train_model(estimators)

# Definindo os parametros
targ_shape = (64, 64)
dataset_name = 'amazon_data_%s.npz' % (targ_shape[0])
estimators = 500

# Loading the dataset
Xtr, Xval, ytr, yval = load_dataset(dataset_name)
create_train_model(estimators)

