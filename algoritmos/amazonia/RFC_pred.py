# Fazendo previsoes separadamente
from utils import *
rfc = joblib.load(base_dir+'/'+'RFC_128.sav')

Xtr, Xte, Xval, ytr, yte, yval = load_dataset()
resultado = rfc.score(Xte, yte)
print(resultado)
prev_val = evaluation(rfc, Xval, yval)
# Test set
prev_te = evaluation(rfc, Xte, yte)
print('Amazon Dataset: ', targ_shape)
print('F1_score_validation: ', prev_val)
print('F1_score_test: ', prev_te)