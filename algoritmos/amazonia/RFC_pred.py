# Fazendo previsoes separadamente
from utils import *


# Definindo os parametros
targ_shape = (16,16,3)
targ_size = targ_shape[:-1]
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])

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


# Carregando o modelo
rfc = joblib.load(base_dir+'/'+'RFC_%s.sav'%(targ_shape[0]))
# Carregando o testset inteiro
Xte, yte = load_testset(dataset_name)
# Classify only one image
image_no = 40477
# Carregando a imagem de test
img_name = 'train_%s.jpg'%image_no
print(img_name)
img = load_img(train_dir+'/'+img_name, target_size=targ_size)
imgarray = img_to_array(img)
imgarray = imgarray.reshape((1,)+imgarray.shape) # Alterando a dimensão, agora é um vetor unidimensional
imgarray = imgarray/255

# Avaliando o modelo
resultado = rfc.score(Xte, yte)
print(resultado)

print('Amazon Dataset: ', targ_shape)
print('Score_test: ', resultado)