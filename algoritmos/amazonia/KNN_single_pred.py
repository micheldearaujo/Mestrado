# Fazendo previsoes separadamente com algoritmo K-Nearest Neighbors, com K = 19

from utils import *


# Definindo os parametros
targ_shape = (8, 8, 3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
sample_size = 404  #len(test_fnames)*0.1

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
knn = joblib.load(base_dir+'/'+'KNN_%s.sav'%targ_shape[0])

# Carregando o testset inteiro
#Xte, yte = load_testset(dataset_name)

# Test the algorithm in 404 randomly sampled images from the test directory
# To get the average classifying time

classifying_times = []
for k in range(sample_size):
    imagefile = test_fnames[np.random.randint(0, len(test_fnames))]
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

    # Predicting the label
    start_time=time.monotonic()
    predicted_labels = knn.predict(imgarray)
    end_time=time.monotonic()
    tempo = timedelta(seconds=end_time - start_time)
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
    knn_df = pd.DataFrame(all_labels, columns=['Labels'])
    knn_df['True_labels'] = pd.Series(true_labels_list)
    knn_df['Predicted_labels'] = pd.Series(predicted_labels[0])
    print(knn_df)
    print('As classes previstas da imagem são: ')
    print(knn_df[knn_df['Predicted_labels']==1]['Labels'])
    print('\n')
    print('Tempo de Classificação:')
    print(tempo)
    classifying_times.append(tempo)

file=open(base_dir+'/'+'KNN_ClassificationTime.txt','a')
file.write('Image Size: %s\n'%targ_shape[0])
file.write('Average Single prediction time: %s\n'%np.mean(classifying_times))
#file.write('Standard deviation Single prediction time: %s\n'%np.std(classifying_times))
file.write('----------------------------------------------------\n')
file.close()