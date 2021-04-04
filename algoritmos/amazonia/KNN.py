import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as cm, classification_report as cr, f1_score
import joblib

start_time = time.monotonic()
# Definindo o caminho dos diretorios
#base_dir = '/home/michel/PycharmProjects/data/amazonia' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Definindo os parametros
targ_shape = (64,64)
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])


# Para carregar os dados
def load_dataset():
    data = np.load(base_dir+'/'+dataset_name)
    X, y = data['arr_0'], data['arr_1']
    print('Dimensões: ')
    print('X: ',X.shape, '\ny: ', y.shape)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    Xtr = Xtr.reshape(Xtr.shape[0], targ_shape[0] * targ_shape[0] * 3)  ## Vamos concatenar os dados das 3 dimensoes em apenas 1 dimensão
    Xte = Xte.reshape(Xte.shape[0], targ_shape[0] * targ_shape[0] * 3)
    # Dividindo o set test em dois, para temos validation+test
    Xval = Xte[:4048, :]
    yval = yte[:4048]
    Xte = Xte[4048:, :]
    yte = yte[4048:]
    # Normalizando os dados entre 0 e 1
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.fit_transform(Xval)
    Xte = scaler.fit_transform(Xte)
    return Xtr, Xte, Xval, ytr, yte, yval

def evaluation(x, true):
    ypred = knn.predict(x)
    return f1_score(true, ypred, average='samples')

# Criando o modelo

Xtr, Xte, Xval, ytr, yte, yval = load_dataset()

# error_rate = []
# Escolhendo o melhor k
# for i in range(1, 41):
#     print('loop %s' % i)
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(Xtr, ytr)
#     pred_i = knn.predict(Xval)
#     error_rate.append(np.mean(pred_i != yval))
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 41), error_rate, color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(Xtr, ytr)

# Realizando previsoes e avaliando a validação
# Validation set
prev_val = evaluation(Xval, yval)
score_val = knn.score(Xval, yval)
# Test set
prev_te = evaluation(Xte, yte)
score_te = knn.score(Xte, yte)
# Salvando o modelo
filename = 'KNN_%s.sav'%targ_shape[0]
joblib.dump(knn, base_dir+'/'+filename)


print('Amazon Dataset: ', targ_shape)
print('F1_score_validation: ', prev_val)
print('F1_score_test: ', prev_te)
print('Score_validation: ', score_val)
print('Score_test: ', score_te)
end_time = time.monotonic()
print('Tempo do treinamento: ')
print('\n')
print(timedelta(seconds=end_time - start_time))