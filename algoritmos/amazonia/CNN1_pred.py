# Prevendo novas imagens

from utils import *

#opt = SGD(lr=0.01, momentum=0.9)
opt = 'adam'
targ_shape = (16,16,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
model_name = 'CNN1_CDA_%s_adam.h5'%(targ_shape[0])

mapping_csv = pd.read_csv(base_dir + '/train_classes.csv')
print("A dimensão do dataframe é: ",mapping_csv.shape) # Dimensões do dataframe com os labels

# Carregando a imagem de test
img_name = 'train_40470.jpg'
img = load_img(train_dir+'/'+img_name, target_size=targ_size)
imgarray = img_to_array(img)
imgarray = imgarray.reshape((1,)+imgarray.shape) # Alterando a dimensão, agora é um vetor unidimensional
imgarray = imgarray/255


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

mapping = mapping(mapping_csv)
labels_map, inv_labels_map = create_tag_map(mapping_csv)
# realizando a previsao da imagem nova
single_predict = modelo.predict_classes(imgarray)
multi_predict = modelo.predict_proba(imgarray)


classes =[]
for i in range(len(inv_labels_map)):
    classes.append(inv_labels_map[i])

df3_labels = pd.DataFrame(classes, columns=['True'])
predicted_proba = pd.Series(multi_predict[0])
df3_labels['Predicted_proba'] = predicted_proba
def define_label(x):
    if x>0.5:
        return 'TRUE'
    else:
        return '-'
df3_labels['Predicted_label'] = df3_labels['Predicted_proba'].apply(define_label)
print(df3_labels)