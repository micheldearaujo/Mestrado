# Modelo com data augmentation  para o treinamento de uma rede neural convolucional
# Com dados de classificação multi-label

# Importanto as bibliotecas necessárias
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import time
from datetime import timedelta

start_time = time.monotonic()

# Definindo o caminho dos diretorios
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Carregamento dos dados já criamos no 'create_dataset.py'
def load_dataset():
    # Carregando
    data = np.load(base_dir + '/amazon_data.npz')
    X, y = data['arr_0'], data['arr_1']
    # Separando os sets de training e testing
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    print('As dimensões dos vetores são: \n')
    print('Xtr: ', Xtr.shape)
    print('\n')
    print('ytr: ', Xtr.shape)
    print('\n')
    print('Xte: ', Xtr.shape)
    print('\n')
    print('yte: ', Xtr.shape)
    print('\n')
    return Xtr, Xte, ytr, yte


# Criando a função para calcular o fbeta score
def fbeta(y_true, y_pred, beta=2):
    # Clipando a previsao
    y_pred = backend.clip(y_pred, 0, 1)
    tp = backend.sum(backend.round(backend.clip(y_true*y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred-y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true-y_pred, 0, 1)), axis=1)
    # Calculando a precisao
    p = tp/(tp+fp+backend.epsilon())
    # Calculando o Recall
    r = tp/(tp+fn+backend.epsilon())
    # calculando o fbeta, tirado a média para cada classe
    bb = beta**2
    fbeta_score = backend.mean((1+bb)*(p*r)/(bb*p+r+backend.epsilon()))
    return fbeta_score


# Criando o modelo de CNN // usaremos o block VGG
def define_model(in_shape=(128,128,3), out_shape=17):
    modelo = Sequential()
    modelo.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
    modelo.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Dropout(0.2))
    modelo.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    modelo.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Dropout(0.2))
    modelo.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    modelo.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Dropout(0.2))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(out_shape, activation='sigmoid'))
    # Compilando
    opt = SGD(lr=0.01, momentum=0.9)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
    return modelo


# Plotando o resultado do treinamento
def resumo(modelohis):
    # Plotando o loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(modelohis.history['loss'], color='blue', label='Training Loss')
    plt.plot(modelohis.history['val_loss'], color='orange', label='Validation Loss')
    plt.legend()
    # Plotando a acurácia
    plt.subplot(212)
    plt.title('Fbeta Score')
    plt.plot(modelohis.history['fbeta'], color='blue', label='Training Fbeta')
    plt.plot(modelohis.history['val_fbeta'], color='orange', label='Validation Fbeta')
    plt.legend()
    # Salvando o gráfico
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot2.png')
    plt.close()


# Rodando o modelo
def run():
    # Load
    Xtr, Xte, ytr, yte = load_dataset()
    # Criando os geradores de data
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       rotation_range=90)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    # Aplicando os iteradores
    train_it = train_datagen.flow(Xtr, ytr, batch_size=128)
    test_it = test_datagen.flow(Xte, yte, batch_size=128)
    # Definindo o modelo
    modelo = define_model()
    # Fitando
    modelohis = modelo.fit_generator(train_it,
                                     steps_per_epoch=len(train_it),
                                     validation_data=test_it,
                                     validation_steps=len(test_it),
                                     epochs=200,
                                     verbose=1, callbacks=[early_stop])
    # Avaliando o modelo
    loss, fbeta = modelo.evaluate_generator(test_it,
                                            steps=len(test_it),
                                            verbose=1)
    print('> loss=%.3f, fbeta=%.3f'%(loss, fbeta))
    # Plotando as curvas de aprendizado
    resumo(modelo)


# Por fim, rodando o modelo

run()
end_time = time.monotonic()
print('Tempo do treinamento: ')
print('\n')
print(timedelta(seconds=end_time - start_time))

# Salvando o modelo para futuras previsoes
modelo.save_model('CNN1_CDA.h5')