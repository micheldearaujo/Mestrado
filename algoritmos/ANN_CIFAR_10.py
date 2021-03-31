# O Objetivo deste programa é utilizar algoritmos de Artificial Neural Networks
# Para realizar classificação do dataset 'CIFAR-10', ou seja, classificação multiclasse.
# O objetivo primordial é entender como algoritmos as ANN funcionam e como usá-las a partir
# de diferentes formas de dados (dados prontos, já em arrays e dados em imagens)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Carregando os dados
# O dataset ja vem separado em dados de treinamento e teste.
# talvez eu faça uma separação para criar um set de validação
(Xtr, ytr), (Xte, yte) = cifar10.load_data()

# Explorando um pouco os dados

print('A dimensao dos dados sao: ')
print('X: ',Xtr.shape)
print('X: ',ytr.shape)
print('X: ',Xte.shape)
print('X: ',yte.shape)

# Uma coisa que precisamos fazer é normalizar os pixels entre 0 e 1, e isso se aplica para todos eles:

Xtr, Xte = Xtr/255.0, Xte/255.

# Agora vamos separar o Xte para termos uma parte para validação

Xval, yval = Xte[:5000], yte[:5000]
Xte, yte = Xte[5000:], yte[5000:]

print('A dimensao dos dados sao: ')
print('X: ',Xtr.shape)
print('X: ',ytr.shape)
print('X: ',Xte.shape)
print('X: ',yte.shape)
print('X: ',Xval.shape)
print('X: ',yval.shape)

# Nice, agora vamos criar nosso modelo de ANN

modelo = models.Sequential()
modelo.add(layers.Dense(30, activation = 'relu'))
modelo.add(layers.Dropout(rate = 0.5))
modelo.add(layers.Dense(20, activation = 'relu'))
modelo.add(layers.Dropout(rate = 0.5))
modelo.add(layers.Dense(10, activation = 'relu'))
modelo.add(layers.Dropout(rate = 0.5))
modelo.add(layers.Flatten())
modelo.add(layers.Dense(10, activation = 'softmax'))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)
modelo.fit(x=Xtr, y=ytr, epochs=200, validation_data=(Xval, yval))

# vamos adicionar dropouts