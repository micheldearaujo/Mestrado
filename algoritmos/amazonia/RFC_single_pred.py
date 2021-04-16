# Fazendo previsoes separadamente

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend
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
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Definindo os parametros
targ_shape = (64, 64, 3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
estimators = 500
sample_size = 404 #len(test_fnames)*0.1


# Definindo o arquivo csv com os nomes dos arquivos e os labels
mapping_csv = pd.read_csv(base_dir + '/train_classes.csv')
print("A dimensão do dataframe é: ", mapping_csv.shape) # Dimensões do dataframe com os labels

def load_testset(dataset_name):
    # Carregando
    data = np.load(base_dir + '/'+ dataset_name)
    X, y = data['arr_0'], data['arr_1']
    # Criando o testset, lembrando que os primeiros 4048 são de validação, já utilizados em cima
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    Xtr = Xtr.reshape(Xtr.shape[0], targ_shape[0] * targ_shape[0] * 3)  ## Vamos concatenar os dados das 3 dimensoes em apenas 1 dimensão
    Xte = Xte.reshape(Xte.shape[0], targ_shape[0] * targ_shape[0] * 3)
    Xte, yte = Xte[4048:,:], yte[4048:]
    print('As dimensões dos vetores são: \n')
    print('Xte shape: ', Xte.shape)
    print('\n')
    print('yte shape: ', yte.shape)
    print('\n')
    return Xte, yte

def create_tag_map(mapping_csv):
    labels = set()
    for i in range(len(mapping_csv)):
        tags = mapping_csv['tags'][i].split(' ')
        labels.update(tags)

    labels = list(labels)
    labels.sort()
    labels_map = {labels[k]: k for k in range(len(labels))}
    inv_labels_map = {k: labels[k] for k in range(len(labels))}
    return labels_map, inv_labels_map

# Vamos criar um dicionário contendo o nome de todas as imagens e suas respectivas classes
def create_file_mapping(mapping_csv):
    mapping=dict() # Criamos um dicionário vazio
    for j in range(len(mapping_csv)):
        """ percorremos o dataframe inteiro, pegando cada nome da imagem
        e sua respectiva tag, então separamos a tag por espaço
        e dizemos que o nome da tag é igual a sua tag, isso cria o dicionário!
        Agora temos multilabels para cada imagem.
        """
        name, tags = mapping_csv['image_name'][j], mapping_csv['tags'][j]
        mapping[name] = tags.split(' ')
    return mapping




# Carregando o modelo
rfc = joblib.load(base_dir+'/'+'RFC_%s_%s.sav'%(targ_shape[0],estimators))

# Carregando o testset inteiro
#Xte, yte = load_testset(dataset_name)


# Test the algorithm in 404 randomly sampled images from the test directory
# To get the average classifying time
classifying_times = []
for k in range(sample_size):
    imagefile = test_fnames[np.random.randint(0, len(test_fnames))]

    # Carregando a imagem de test
    img_name = imagefile
    print(img_name)
    img = load_img(train_dir+'/'+img_name, target_size=targ_size)
    imgarray = img_to_array(img)
    imgarray = imgarray.reshape(1,-1) # Alterando a dimensão, agora é um vetor unidimensional
    imgarray = imgarray/255

    # Avaliando o modelo
    #score = rfc.score(Xte, yte)
    #print('Amazon Dataset: ', targ_shape)
    #print('Score_test: ', score)

    # Tentando mostrar o resultado da previsao

    start_time = time.monotonic()
    predicted_labels = rfc.predict(imgarray)
    end_time=time.monotonic()
    print(predicted_labels)

    # Let`s get the image True Labels:
    mapping = create_file_mapping(mapping_csv)
    true_labels = mapping[imagefile.split('.')[0]]


    # Criando uma lista com todas as classes possiveis
    labels_map, inv_labels_map = create_tag_map(mapping_csv)
    all_labels = []
    for i in range(len(inv_labels_map)):
        all_labels.append(inv_labels_map[i])


    # Criando uma lista ordenada com as classes verdadeiras e todas as outras classes
    true_labels_list =[0 for i in range(len(all_labels))]
    for class_ in true_labels:
        index_ = all_labels.index(class_)
        true_labels_list[index_] = 1


    # Criando um dataframe para organizar todas as informações da classificacao da imagem
    rfc_df = pd.DataFrame(all_labels, columns=['Labels'])
    rfc_df['True_labels'] = pd.Series(true_labels_list)
    rfc_df['Predicted_labels'] = pd.Series(predicted_labels[0])
    print(rfc_df)


    print('As classes previstas da imagem são: ')
    print(rfc_df[rfc_df['Predicted_labels']==1]['Labels'])
    print('\n')


    tempo = timedelta(seconds=end_time - start_time)
    print('Tempo de Classificação:')
    print(tempo)
    classifying_times.append(tempo)

file=open(base_dir+'/'+'RFC_ClassificationTime.txt','a')
file.write('Image Size: %s_%s\n'%(targ_shape[0],estimators))
file.write('Average Single prediction time: %s\n'%np.mean(classifying_times))
#file.write('Standard deviation Single prediction time: %s\n'%np.std(classifying_times))
file.write('----------------------------------------------------\n')
file.close()