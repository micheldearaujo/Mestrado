# Prevendo novas imagens

from utils import *

opt = SGD(lr=0.01, momentum=0.9)
targ_shape = (16,16,3)
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0])
model_name = 'CNN1_CDA_16_adam.h5'

# Carregando a imagem de test
img_name = 'train_40477.jpg'
img = load_img(train_dir+'/'+img_name, target_size=(16,16))
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

modelo = load_model(base_dir+'/'+model_name, compile=False)
modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
Xte, yte = load_testset(dataset_name)
test_it = test_datagen.flow(Xte, yte, batch_size=targ_shape[0])
loss, fbeta = modelo.evaluate(test_it,
                              steps=len(test_it),
                              verbose=1)