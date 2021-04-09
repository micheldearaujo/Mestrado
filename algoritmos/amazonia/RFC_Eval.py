# Fazendo previsoes separadamente
from utils import *
# -----------------------------------------------
start_time=time.monotonic()
# Definindo os parametros
targ_shape = (8,8,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])

# -------------------------------------------------
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


# ------------------------------------------------------------
# Calling the dictionary function
mapping = create_file_mapping(mapping_csv)

# Criando os dicionários que relacionam um numero com cada classe
labels_map, inv_labels_map = create_tag_map(mapping_csv)

# Criando uma lista com todas as classes possiveis
all_labels = []
for i in range(len(inv_labels_map)):
    all_labels.append(inv_labels_map[i])

# Carregando o modelo
rfc = joblib.load(base_dir+'/'+'RFC_%s_500.sav'%(targ_shape[0]))

# Carregando o testset inteiro
Xte, yte = load_testset(dataset_name)

# Classifying all the images in the test set
TP, FP, TN, FN = 0, 0, 0, 0
for image_no in range(len(Xte), 2*len(Xte)):
    print('progresso: %s de %s'%(image_no,2*len(Xte)))
    # Carregando a imagem de test
    img_name = 'train_%s.jpg'%image_no
    #print(img_name)
    img = load_img(train_dir+'/'+img_name, target_size=targ_size)
    imgarray = img_to_array(img)
    imgarray = imgarray.reshape(1,-1) # Alterando a dimensão, agora é um vetor unidimensional
    imgarray = imgarray/255

    # Predicting the new image
    predicted_labels = rfc.predict(imgarray)
    #print(predicted_labels)

    # Let`s get the image True Labels:
    true_labels = mapping['train_%s'%image_no]

    # Criando uma lista ordenada com as classes verdadeiras e todas as outras classes
    true_labels_list = [0 for i in range(len(all_labels))]
    for label_ in true_labels:
        index_ = all_labels.index(label_)
        true_labels_list[index_] = 1

    # Criando um dataframe para organizar todas as informações da classificacao da imagem
    rfc_df = pd.DataFrame(all_labels, columns=['Labels'])
    rfc_df['True_labels'] = pd.Series(true_labels_list)
    rfc_df['Predicted_labels'] = pd.Series(predicted_labels[0])
    #print(rfc_df)

    # Calculando os TP, FP, TN, FN
    TP += len(rfc_df[(rfc_df['True_labels'] == 1) & (rfc_df['Predicted_labels'] == 1)])
    FP += len(rfc_df[(rfc_df['True_labels'] == 0) & (rfc_df['Predicted_labels'] == 1)])
    TN += len(rfc_df[(rfc_df['True_labels'] == 0) & (rfc_df['Predicted_labels'] == 0)])
    FN += len(rfc_df[(rfc_df['True_labels'] == 1) & (rfc_df['Predicted_labels'] == 0)])

# definindo e calculando as métricas:
# print('Avg True Positives: ', TP)
# print('Avg False Positives: ', FP)
# print('Avg True Negatives: ', TN)
# print('Avg False Negatives: ', FN)
# Precision
precision = round(TP / (TP + FP), 3)
print('Avg Precision: ', precision)
# Recall (Sensibilidade ou True Positive Rate)
recall = round(TP / (TP + FN), 3)
print('Avg Recal: ', recall)
# F1 Score (Media ponderada entre precision e recall)
f1_score = round(2 * (precision * recall) / (precision + recall), 3)
print('Avg F1_Score:', f1_score)

print('As classes previstas da imagem são: ')
print(rfc_df[rfc_df['Predicted_labels']==1]['Labels'])
print('\n')


#----------------------------------------------------
end_time=time.monotonic()
tempo = timedelta(seconds=end_time - start_time)
print('Tempo de Classificação:')
print(tempo)

file=open('RFC_Scores.txt','a')
file.write('Image Size: %s\n'%targ_shape[0])
file.write('Evaluation time: %s\n'%tempo)
file.write('Avg Precision: %s\n'%precision)
file.write('Avg Recall: %s\n'%recall)
file.write('Avg F1_Score: %s\n'%f1_score)
file.write('----------------------------------------------------\n')
file.close()


### Criar um arquivo para salvar todas as informações já que
# Vou ter que re-treinar todos os RFC