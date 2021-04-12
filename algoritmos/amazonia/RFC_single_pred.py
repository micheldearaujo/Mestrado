# Fazendo previsoes separadamente
from utils import *

start_time=time.monotonic()
# Definindo os parametros
targ_shape = (16, 16,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
estimators=500

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

# Classify only one image
image_no = 40469

# Carregando a imagem de test
img_name = 'train_%s.jpg'%image_no
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

predicted_labels = rfc.predict(imgarray)
print(predicted_labels)

# Let`s get the image True Labels:
mapping = create_file_mapping(mapping_csv)
true_labels = mapping['train_%s'%image_no]


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

end_time=time.monotonic()
tempo = timedelta(seconds=end_time - start_time)
print('Tempo de Classificação:')
print(tempo)

file=open(base_dir+'/'+'RFC_ClassificationTime.txt','a')
file.write('Image Size: %s_%s\n'%(targ_shape[0],estimators))
file.write('Single prediction time: %s\n'%tempo)
file.write('----------------------------------------------------\n')
file.close()