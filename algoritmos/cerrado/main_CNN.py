# Import libraries
from utils import *
from architectures import CNN_model2
import os

# Defining parameters
method = 'ef'
date = 2
patch_size = 15
stride = patch_size//2
px_central = patch_size//2
label_c = 1
label_nc = 0
rows = patch_size
cols = patch_size
times = 10

#%% Creation of irectory to save the model andprob maps
# Parent Directory path 
parent_dir = '/mnt/DATA/Mabel/Doctorado/Deforestation_SAR/Results_paper_IEEE/GitHub/'
dirName = os.path.join(parent_dir, method+'_'+str(date)+'dates')
dirModel = os.path.join(dirName, 'models')
dirProbs = os.path.join(dirName, 'prob_maps')
try:
    # Create target Directory
    os.mkdir(dirName)
    os.mkdir(dirModel)
    os.mkdir(dirProbs)
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
    print("Directory " , dirModel ,  " already exists")
    print("Directory " , dirProbs ,  " already exists")

#%% Load data
root_path = 'Images/'

img_2018_07_21 = load_SAR_image(root_path+'2018_07_21'+'.tif').astype(np.float32)
img_2019_08_09 = load_SAR_image(root_path+'2019_08_09'+'.tif').astype(np.float32)

image_array1 = np.concatenate((img_2018_07_21, img_2019_08_09), axis = -1).astype(np.float32)

h_, w_, channels = image_array1.shape

# Normalization
type_norm = 1
image_array = normalization(image_array1, type_norm)
image_array = image_array[:5120, :5888, :]

# Reference
reference_2019 = np.load('reference_2019.npy')

# Tiles mask 
mask_tiles = np.load('mask_tiles.npy')

mask_amazon = np.zeros((mask_tiles.shape))
tiles_tr = [1, 3, 4, 7, 9, 10, 13, 16]
tiles_val = [6, 12]
tiles_ts = [2, 5, 8, 11, 14, 15]

# Training and validation mask
for tr_ in tiles_tr:
    mask_amazon[mask_tiles == tr_] = 1

for val_ in tiles_val:
    mask_amazon[mask_tiles == val_] = 2

# Test mask
mask_amazon_ts_ = np.zeros((mask_tiles.shape))
for ts_ in tiles_ts:
    mask_amazon_ts_[mask_tiles == ts_] = 1

# Patch extraction
patches_img, labels_c_nc, labels_tr_ts_val = extract_patches_CNN(image_array, reference_2019, mask_amazon, patch_size, stride)

# Training patches
set_train = 1
patches_train_c, patches_train_ref_c, patches_train_nc, patches_train_ref_nc, tr, trc, trnc = get_patches_class_CNN(patches_img, labels_c_nc, labels_tr_ts_val, set_train)

# Validation patches
set_val = 2
patches_val_c, patches_val_ref_c, patches_val_nc, patches_val_ref_nc, vl, vlc, vlnc = get_patches_class_CNN(patches_img, labels_c_nc, labels_tr_ts_val, set_val)

print('amount samples train ', tr)
print('amount samples train change', trc)
print('amount samples train no change', trnc)
print('amount samples validation', vl)
print('amount samples validation change', vlc)
print('amount samples validation no change', vlnc)


#%% Loop Training CNN
tr_time = []

for tm in range(0,times):

    patches_train, patches_train_ref1 = balance_data_CNN(patches_train_c, label_c, patches_train_nc, label_nc, number_class = 2)
    patches_val, patches_val_ref1 = balance_data_CNN(patches_val_c, label_c, patches_val_nc, label_nc, number_class = 2)
    
    # Define model
    model = CNN_model2()
    # Define optimizer
    adam = Adam(lr = 1e-5 , beta_1=0.9)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model_name = 'ef_'+str(tm)+'.h5'
    # Define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(dirModel+'/'+model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]
    # Train the model
    start = time.time()
    model_info = model.fit(patches_train, patches_train_ref1, batch_size=8, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= (patches_val,patches_val_ref1))
    end_training = time.time() - start
    print ('Model took %0.2f seconds to train'%(end_training))
    tr_time.append(end_training)
    del patches_train, patches_train_ref1, patches_val, patches_val_ref1, model

#%% Loop Prediction CNN

ts_time = []
for tm in range(0,times):
    print('time:  ', tm)
    filepath = 'models/ef_'+str(tm)+'.h5'
    model = load_model(dirModel+'/'+model_name)

    npad = ((px_central, px_central), (px_central, px_central), (0, 0))
    image1_pad = np.pad(image_array, pad_width=npad, mode='reflect')
    
    mask_1 = np.ones((mask_amazon_ts_.shape))
    npad_1 = ((px_central, px_central), (px_central, px_central))
    mask_1_pad = np.pad(mask_1, pad_width=npad_1, mode = 'constant',constant_values=(0,0))
    
    [rows_test, cols_test] = np.where(mask_1_pad==1)
    rows_test.astype(np.float32)
    cols_test.astype(np.float32)
    start_test = time.time()
    
    batch = 100000
    n_batch = len(rows_test)//batch
    pred_batch = []
    true_batch = []
    for i in range(0,n_batch+1):
        print(i)
        if (i==n_batch):
            batch_f = len(rows_test)-n_batch*batch
            patches_batch = get_patches_batch_CNN(image1_pad, rows_test[i*batch:], cols_test[i*batch:], px_central, batch_f)
        else: 
            patches_batch = get_patches_batch_CNN(image1_pad, rows_test[i*batch: (i+1)*batch], cols_test[i*batch: (i+1)*batch], px_central, batch)
        pred_ = model.predict(patches_batch)
        pred_batch.append(pred_)
        del patches_batch
       
    end_test = time.time() - start_test
    ts_time.append(end_test)
    pre_final_1 = np.concatenate(pred_batch)
    pre_final_ = pre_final_1[:,1]
    # probability map
    h,w = mask_amazon.shape
    # Reconstructed map
    reconstructed_prob = np.zeros((h,w)).astype('float32')
    t = 0
    for i in range(h):
        for j in range(w):
            #reconstructed[i,j] = result1[t]
            reconstructed_prob[i,j] = pre_final_[t]
            t = t+1
    np.save(dirProbs+'/'+'prob_'+str(tm)+'.npy',reconstructed_prob)    



