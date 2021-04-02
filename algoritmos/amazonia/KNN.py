import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as cm, classification_report as cr

start_time = time.monotonic()
# Definindo o caminho dos diretorios
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Definindo os parametros
targ_shape = (8,8)
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])


# Para carregar os dados
def load_dataset():
    data = np.load(base_dir+'/'+dataset_name)
    X, y = data['arr_0'], data['arr_1']
    print('Dimensões: ')
    print('X: ',X.shape, '\n y: ', y.shape)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr = Xtr.reshape(Xtr.shape[0], targ_shape[0] * targ_shape[0] * 3)  ## Vamos concatenar os dados das 3 dimensoes em apenas 1 dimensão
    Xte = Xte.reshape(Xte.shape[0], targ_shape[0] * targ_shape[0] * 3)
    # Dividindo o set test em dois, para temos validation+test
    Xval = Xte[:6072, :]
    yval = yte[:6072]
    Xte = Xte[6072:, :]
    yte = yte[6072:]
    # Normalizando os dados entre 0 e 1
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.fit_transform(Xval)
    Xte = scaler.fit_transform(Xte)
    return Xtr, Xte, Xval, ytr, yte, yval

# Criando o modelo
def create_model():
    # Criando a instancia e fitando
    knnmodel = KneighborsClassifier(n_neighbors=1)
    knnmodel.fit(Xtr, ytr)
    # Escolhendo o melhor k
    # error_rate = []
    # for i in range(1, 40):
    #     print('loop %s' % i)
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(Xtr, ytr)
    #     pred_i = knn.predict(Xval)
    #     error_rate.append(np.mean(pred_i != yval))

# Realizando previsoes e avaliando a validação
def evaluation(prev):
    ypred = knnmodel.predict(prev)
    print('Classification Report: ')
    print(cr(prev, yte))
    print('Confusion Matrix: ')
    print(cm(prev, yte))
    return prev


Xtr, Xte, Xval, ytr, yte, yval = load_dataset()
create_model()
# Validation set
prev_val = evaluation(Xval)
# Test set
prev_te = evaluation(Xte)
end_time = time.monotonic()
print('Tempo do treinamento: ')
print('\n')
print(timedelta(seconds=end_time - start_time))