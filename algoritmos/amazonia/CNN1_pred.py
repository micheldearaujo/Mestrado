# Prevendo novas imagens

import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import SGD
import time
from datetime import timedelta

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle' # Windows
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Parâmetros do modelo
opt = SGD(lr=0.01, momentum=0.9)
#opt = 'adam'
targ_shape = (16,16,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
model_name = 'CNN1_CDA_%s_adam.h5'%(targ_shape[0])

# Definindo o arquivo csv com os nomes dos arquivos e os labels
mapping_csv = pd.read_csv(base_dir + '/train_classes.csv')
print("A dimensão do dataframe é: ",mapping_csv.shape) # Dimensões do dataframe com os labels






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

# Vamos criar um dicionário relacionando um valor numérico para cada classe string que está no dataset
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

modelo = load_model(base_dir+'/'+model_name, compile=False)
modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
# test_datagen = ImageDataGenerator(rescale=1.0/255.0)
# Xte, yte = load_testset(dataset_name)
# test_it = test_datagen.flow(Xte, yte, batch_size=targ_shape[0])
# loss, fbeta = modelo.evaluate(test_it,
#                               steps=len(test_it),
#                               verbose=1)

# Chamando o dicionario com filenames e classes
mapping = create_file_mapping(mapping_csv)

# Carregando a imagem de test
k=40470
img_name = 'train_%s.jpg'%k
img = load_img(train_dir+'/'+img_name, target_size=targ_size)
imgarray = img_to_array(img)
imgarray = imgarray.reshape((1,)+imgarray.shape) # Alterando a dimensão, agora é um vetor unidimensional
imgarray = imgarray/255

# Criando os dicinários encondes
labels_map, inv_labels_map = create_tag_map(mapping_csv)
# realizando a previsao da imagem nova
single_predict = modelo.predict_classes(imgarray)
multi_predict = modelo.predict_proba(imgarray)

# Criando um dataframe para mostrar as classes, as classes da imagem especifica e as classes previstas

true_classes = mapping['train_%s'%k]
classes =[]
for i in range(len(inv_labels_map)):
    classes.append(inv_labels_map[i])

true_classes_list =['-' for i in range(len(classes))]
for class_ in true_classes:
    index_ = classes.index(class_)
    true_classes_list[index_] = class_

df_labels = pd.DataFrame(classes, columns=['Classes'])
df_labels['True_labels'] = pd.Series(true_classes_list)
df_labels['Predicted_proba'] = pd.Series(multi_predict[0])


# Definindo como TRUE as classes que possuem probabilidade maior que 50%
def define_label(x):
    if x>0.3:
        return 'TRUE'
    else:
        return '-'
df_labels['Predicted_label'] = df_labels['Predicted_proba'].apply(define_label)
print(df_labels)

# Calculando os TP, FP, TN, FN
TP = len(df_labels[(df_labels['True_labels'] != '-') & (df_labels['Predicted_label'] != '-')])
FP = len(df_labels[(df_labels['True_labels'] == '-') & (df_labels['Predicted_label'] != '-')])
TN = len(df_labels[(df_labels['True_labels'] == '-') & (df_labels['Predicted_label'] == '-')])
FN = len(df_labels[(df_labels['True_labels'] != '-') & (df_labels['Predicted_label'] == '-')])
print(TP+TN+FP+FN)

