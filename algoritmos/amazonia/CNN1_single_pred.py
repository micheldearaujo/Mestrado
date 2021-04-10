# Prevendo novas imagens

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import SGD
import time
from datetime import timedelta

start_time = time.monotonic()

# Definindo o caminho dos diretorios
#base_dir = '/home/michel/data/amazonia/kaggle' # Ubuntu
base_dir = 'D:/michel/data/amazonia/kaggle' # Windows
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Parâmetros do modelo
opt = SGD(lr=0.01, momentum=0.9)
targ_shape = (32,32,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
model_name = 'CNN1_CDA_%s_SGD.h5'%(targ_shape[0])

# Definindo o arquivo csv com os nomes dos arquivos e os labels
mapping_csv = pd.read_csv(base_dir + '/train_classes.csv')
print("A dimensão do dataframe é: ", mapping_csv.shape) # Dimensões do dataframe com os labels


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
# Criaremos uma funcao para carregar os dados de test, mas eh so pra ter quantas imagens sao
def load_testset(dataset_name):
    # Carregando
    data = np.load(base_dir + '/'+ dataset_name)
    X, y = data['arr_0'], data['arr_1']
    # Criando o testset, lembrando que os primeiros 4048 são de validação, já utilizados em cima
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    Xte, yte = Xte[4048:,:], yte[4048:]
    print('As dimensões dos vetores são: \n')
    print('Xte shape: ', Xte.shape)
    print('\n')
    print('yte shape: ', yte.shape)
    print('\n')
    return Xte, yte

# Carregando e compilando o modelo já treinado
modelo = load_model(base_dir+'/'+model_name, compile=False)
modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
# Chamando o dicionario com filenames e classes
mapping = create_file_mapping(mapping_csv)
# Criando os dicionários que relacionam um numero com cada classe
labels_map, inv_labels_map = create_tag_map(mapping_csv)
# Carregando o testset
Xte, yte = load_testset(dataset_name)

# Criando uma lista com todas as classes possiveis
classes = []
for i in range(len(inv_labels_map)):
    classes.append(inv_labels_map[i])

# Definindo o threshold (Tolerancia para classficiar como sim ou nao)
threshold = 0.3
# Classify only one image
image_no = 40477

# Carregando a imagem de test
img_name = 'train_%s.jpg'%image_no
print(img_name)
img = load_img(train_dir+'/'+img_name, target_size=targ_size)
imgarray = img_to_array(img)
imgarray = imgarray.reshape((1,)+imgarray.shape) # Alterando a dimensão, agora é um vetor unidimensional
imgarray = imgarray/255

# realizando a previsao da imagem nova
prediction = modelo.predict_proba(imgarray)

# Criando uma lista com as classes verdadeiras da referida imagem
true_classes = mapping['train_%s'%image_no]

# Criando uma lista ordenada com as classes verdadeiras e todas as outras classes
true_classes_list =[0 for i in range(len(classes))]
for class_ in true_classes:
    index_ = classes.index(class_)
    true_classes_list[index_] = 1

# Criando um dataframe para organizar todas as informações da classificacao da imagem
df_labels = pd.DataFrame(classes, columns=['Labels'])
df_labels['True_labels'] = pd.Series(true_classes_list)
df_labels['Predicted_proba'] = pd.Series(prediction[0])

# Definindo como 1 as classes que possuem probabilidade maior que % e 0 o contrario
def enconder(probabilidade):
    if probabilidade>threshold:
        return 1
    else:
        return 0
df_labels['Predicted_labels'] = df_labels['Predicted_proba'].apply(enconder)
print(df_labels)

TP = len(df_labels[(df_labels['True_labels'] == 1) & (df_labels['Predicted_labels'] == 1)])
FP = len(df_labels[(df_labels['True_labels'] == 0) & (df_labels['Predicted_labels'] == 1)])
TN = len(df_labels[(df_labels['True_labels'] == 0) & (df_labels['Predicted_labels'] == 0)])
FN = len(df_labels[(df_labels['True_labels'] == 1) & (df_labels['Predicted_labels'] == 0)])
print('True Positives: ',TP)
print('False Positives: ',FP)
print('True Negatives: ',TN)
print('False Negatives: ',FN)

# definindo e calculando as métricas
# Precision
precision = round(TP / (TP + FP), 3)
print('Avg Precision: ', precision)

# Recall (Sensibilidade ou True Positive Rate)
recall = round(TP / (TP + FN), 3)
print('Avg Recal: ', recall)

# F1 Score (Media ponderada entre precision e recall)
f1_score = round(2 * (precision * recall) / (precision + recall), 3)

# Overall Accuracy (Porcentagem de acertos sobre o total)
acc = round((TP+TN)/(TP + FP + TN + FN), 3)

print('Avg Accuracy: ', acc)
print('Avg F1_Score:', f1_score)
print('\n')
print('As classes previstas da imagem são: ')
print(df_labels[df_labels['Predicted_labels']==1]['Labels'])
print('\n')
# Terminando a contagem do tempo
end_time = time.monotonic()
print('Tempo de Classificação: ')
print(timedelta(seconds=end_time - start_time))