# Prevendo novas imagens

from utils import *

# Definindo o caminho dos diretorios
base_dir = 'D:/michel/data/amazonia/kaggle'
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

model_name = 'CNN1_CDA_16_adam.h5'
modelohis = load_model(base_dir+'/'+model_name, compile=False)
