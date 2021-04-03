from utils import *
from architectures import unet, build_enconder, siamese_model
import os

#%% Method (U-Net-> 1, FC-Siamese ->2)
method = 'unet'
date = 2
if method == 'unet':
    method_ = 1
    print('METHOD: U-Net')
if method == 'fc-siam':
    method_ = 2
    print('METHOD: FC-Siamese')

# Parameters
patch_size = 128
stride = patch_size//4
# Percent of deforestation class
percent = 20
number_class = 3
weights = [0.075, 0.925, 0]
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
band = channels//2

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


#%% Training loop

time_tr = []
for tm in range(0,times):
    print('time: ', tm)
    # Trainig tiles
    patches_tr, patches_tr_ref = patch_tiles(tiles_tr, mask_tiles, image_array, reference_2019, patch_size, stride)
    patches_tr_aug, patches_tr_ref_aug = bal_aug_patches(percent, patch_size, patches_tr, patches_tr_ref)
    patches_tr_ref_aug_h = tf.keras.utils.to_categorical(patches_tr_ref_aug, number_class)
    del patches_tr, patches_tr_ref
    
    # Validation tiles
    patches_val, patches_val_ref = patch_tiles(tiles_val, mask_tiles, image_array, reference_2019, patch_size, stride)
    patches_val_aug, patches_val_ref_aug = bal_aug_patches(percent, patch_size, patches_val, patches_val_ref)
    patches_val_ref_aug_h = tf.keras.utils.to_categorical(patches_val_ref_aug, number_class)
    del patches_val, patches_val_ref 
    
    rows = patch_size
    cols = patch_size
    adam = Adam(lr = 1e-5 , beta_1=0.9)
    batch_size = 8
    
    loss = weighted_categorical_crossentropy(weights)
    if method_ == 1:
        model = unet((rows, cols, channels), number_class)
    if method_ == 2:
        model = siamese_model((rows, cols, band))

    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    # print model information
    model.summary()
    model_name = method+str(tm)+'.h5'
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min') ---- val_accuracy
    checkpoint = ModelCheckpoint(dirModel+'/'+model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]
    # train the model
    start_training = time.time()
    # For U-Net
    if method_ == 1:
        model_info = model.fit(patches_tr_aug, patches_tr_ref_aug_h, batch_size=batch_size, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= (patches_val_aug, patches_val_ref_aug_h) )
    # For FC Siamese
    if method_ == 2:
        model_info = model.fit([patches_tr_aug[:,:,:,:band],patches_tr_aug[:,:,:,band:]], patches_tr_ref_aug_h, batch_size=batch_size, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= ([patches_val_aug[:,:,:,:band],patches_val_aug[:,:,:,band:]], patches_val_ref_aug_h) )
    end_training = time.time() - start_training
    time_tr.append(end_training)

#%% Test Siamsese
time_ts = []
for tm in range(0,times):
    print('time: ', tm)
    model = load_model(dirModel+'/'+model_name, compile=False)
    
    #Unet
    if method_ == 1:
        prob_recontructed, end_test = output_prediction_FC(model, image_array, reference_2019, patch_size)
        
    #Siamese
    if method_ == 2:
        start_test = time.time()
        patch_ts = patches_with_out_overlap(image_array, patch_size, img_type = 2)
        prediction =  model.predict([patch_ts[:,:,:,:band], patch_ts[:,:,:,band:]]).astype(np.float32)
        pred1 = prediction[:,:,:,1]
        end_test =  time.time() - start_test
        prob_recontructed = pred_recostruction(patch_size, pred1, reference_2019)
    
    np.save(dirProbs+'/'+'prob_'+str(tm)+'.npy',prob_recontructed) 

    time_ts.append(end_test)


    

