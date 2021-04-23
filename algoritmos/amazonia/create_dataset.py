
# Importando as bibliotecas necessarias
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from datetime import timedelta
import time

# Definindo o caminho dos diretorios
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)
targ_shape = (96, 96, 3)
targ_size = targ_shape[:-1]

# Plotando algumas imagens
def plot_imagens():
    for i in range(9):
        plt.subplot(330+1+i)
        filename = train_dir + '/' +train_fnames[i+1]
        image = imread(filename)
        plt.imshow(image)
    plt.show()
#plot_imagens()

# Criando mapas
# Como vimos, as figuras não estão legendadas (no nome ou por pastas, por exemplo). Em vez disso,
# temos um arquivo csv separado com nomes das images e o respectivo label. Dessa forma, precisamos
# criar um mapa para atribuir os labels do .csv para as imagems!!

mapping_csv = pd.read_csv(base_dir + '/train_classes.csv')
print("A dimensão do dataframe é: ",mapping_csv.shape) # Dimensões do dataframe com os labels

# Vamos criar um conjunto (set) de tags para cada atributo na coluna tags.
# Como uma figura pode ter mais de uma classificação, então iremos atribuir uma lista de classificações para cada imagem.
# Dessa forma, precisamos transformar as str da tag em numeros
# Usamos o set() pois nele contem apenas valores unicos!

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

# Criando um mapeamento dos nomes dos arquivos para as tags
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

# Carregando as images!
# Agora podemos enfim tentar carregar as imagens (Se a RAM conseguir).
# Vamos diminuir o tamanho das imagens (para 128), para facilitar e transformar seus valores em 8 bits (0 a 255 o valor do pixel).


# Criando one-hot-enconde:
def one_hot_enconde(tags, mapping):
    enconding = np.zeros(len(mapping), dtype='uint8')
    for tag in tags:
        enconding[mapping[tag]] = 1
    return enconding


# Criando uma função para carregar as imagens
def load_dataset(path, file_mapping, tag_mapping,targ_size=(128,128)):
    pics, targets = list(), list()
    # Vai percorrer todas as imagens no diretório
    for filename in os.listdir(path):
        # Carrega a imagen com as dimensoes especificadas
        pic = load_img(path + '/' + filename, target_size=targ_size)
        # Transforma a imagens numa matriz de valores dos pixels
        pic = img_to_array(pic, dtype='uint8')
        # Cria o mapeamento das tags para o nome dos arquivos
        tags = file_mapping[filename[:-4]]
        # Transforma a tag de texto para numeros
        target = one_hot_enconde(tags, tag_mapping)
        # Junta todos os arrays criados nas listas de imagens e labels
        pics.append(pic)
        targets.append(target)
    # as imagens e os labels eram listas, vamos carregar como arrays
    X = np.asarray(pics, dtype='uint8')
    y = np.asarray(targets, dtype='uint8')
    return X, y

# Executando


# definindo o nome do csv
filename='train_classes.csv'
# Definindo o nome dos dados a serem salvos
dataset_name = 'amazon_data_%s.npz'%(targ_size[0])
# Criando o dataframe com as tags das imagens
mapping_csv = pd.read_csv(base_dir + '/' +filename)
# Criando o dicionário com as tags strings para numeros
tag_mapping, _ = create_tag_map(mapping_csv)
# Criando o mapa dos arquivos para as tags list
file_mapping = create_file_mapping(mapping_csv)
# Carregando as iamgens jpegs em arrays
X, y = load_dataset(train_dir, file_mapping, tag_mapping, targ_size)
print(X.shape, y.shape)
# Salvando os arrays, só precisamos usar este código 1 vez para
# para cada dataset
np.savez_compressed(base_dir + '/' + dataset_name, X, y)

# Para carregar os dados
# data = np.load(base_dir +'/amazon_data.npz')
# X, y = data['arr_0'], data['arr_1']
