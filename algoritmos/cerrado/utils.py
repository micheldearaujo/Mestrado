# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from osgeo import ogr, gdal
import tensorflow as tf
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import skimage.morphology 
from skimage.morphology import disk
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Conv2DTranspose, Dropout, concatenate, \
Input, UpSampling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add, ZeroPadding2D, DepthwiseConv2D, \
AveragePooling2D, Lambda
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# Functions

def load_tiff_image(image):
    '''Function to read tif images'''
    print (image)
    gdal_header = gdal.Open(image)
    img = gdal_header.ReadAsArray()
    return img

def load_SAR_image(image):
    '''Function to read SAR images'''
    print (image)
    gdal_header = gdal.Open(image)
    db_img = gdal_header.ReadAsArray()
    db_img = np.transpose(db_img, (1, 2, 0))
    temp_db_img = 10**(db_img/10)
    temp_db_img[temp_db_img>1] = 1
    return temp_db_img

def data_augmentation(image, labels):
    '''Generate more syntetic data using data augmentation'''
    aug_imgs = np.zeros((5, image.shape[0], image.shape[1], image.shape[2]))
    aug_lbs = np.zeros((5, image.shape[0], image.shape[1]))
    for i in range(0, len(aug_imgs)):
        aug_imgs[0, :, :, :] = image
        aug_imgs[1, :, :, :] = np.rot90(image, 1)
        aug_imgs[2, :, :, :] = np.rot90(image, 2)
        aug_imgs[3, :, :, :] = np.flip(image,0)
        aug_imgs[4, :, :, :] = np.flip(image, 1)
    for i in range(0, len(aug_lbs)):
        aug_lbs[0, :, :] = labels
        aug_lbs[1, :, :] = np.rot90(labels, 1)
        aug_lbs[2, :, :] = np.rot90(labels, 2)
        aug_lbs[3, :, :] = np.flip(labels,0)
        aug_lbs[4, :, :] = np.flip(labels, 1)
    return aug_imgs, aug_lbs

def normalization(image, norm_type = 1):
    '''Normalization of data. 1->StandardScaler, 2->MinMaxScaler, 3->MinMaxScaler'''
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
      scaler = StandardScaler()
    if (norm_type == 2):
      scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
      scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1

def extract_patches(input_image, reference,  patch_size, stride):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))

    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)

    return patches_array, patches_ref

def patch_tiles(tiles, mask_amazon, image_array, image_ref, patch_size, stride):
    '''Extraction of image patches and labels '''
    patches_out = []
    label_out = []
    for num_tile in tiles:
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        
        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        patches_out.append(patches_img)
        label_out.append(patch_ref)
        
    patches_out = np.concatenate(patches_out)
    label_out = np.concatenate(label_out)
    return patches_out, label_out

def bal_aug_patches(percent, patch_size, patches_img, patches_ref):
    '''Apply data augmentation if the patch contains more of (x%) of deforestation'''
    patches_images = []
    patches_labels = []
    
    for i in range(0,len(patches_img)):
        patch = patches_ref[i]
        class1 = patch[patch==1]
        
        if len(class1) >= int((patch_size**2)*(percent/100)):
            patch_img = patches_img[i]
            patch_label = patches_ref[i]
            img_aug, label_aug = data_augmentation(patch_img, patch_label)
            patches_images.append(img_aug)
            patches_labels.append(label_aug)
    
    patches_bal = np.concatenate(patches_images).astype(np.float32)
    labels_bal = np.concatenate(patches_labels).astype(np.float32)
    return patches_bal, labels_bal

def patches_with_out_overlap(img, stride, img_type):
    '''Extract patches without overlap to test models, img_type = 1 (reference image), img_type = 2 (images)'''
    if img_type == 1:
        h, w = img.shape 
        num_patches_h = int(h/stride)
        num_patches_w = int(w/stride)
        patch_t = []
        counter=0
        for i in range(0,num_patches_w):
            for j in range(0,num_patches_h):
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1)]
                counter=counter+1
                patch_t.append(patch)
        patch_t1=np.asarray(patch_t)
        
    if img_type == 2:
        h, w, c = img.shape 
        num_patches_h = int(h/stride)
        num_patches_w = int(w/stride)
        patch_t = []
        counter=0
        for i in range(0,num_patches_w):
            for j in range(0,num_patches_h):
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1), :]
                counter=counter+1
                patch_t.append(patch)
        patch_t1=np.asarray(patch_t)
    
    return patch_t1

def test_FCN(net, patch_test):
    ''' Function to test FCN model'''
    predictions = net.predict(patch_test)
    print(predictions.shape)
    pred1 = predictions[:,:,:,1]
    p_labels=predictions.argmax(axis=-1)
    return p_labels, pred1

def pred_recostruction(patch_size, pred_labels, image_ref):
    ''' Reconstruction of whole prediction image'''
    stride = patch_size
    h, w = image_ref.shape
    num_patches_h = int(h/stride)
    num_patches_w = int(w/stride)
    count = 0
    img_reconstructed = np.zeros((num_patches_h*stride,num_patches_w*stride))
    for i in range(0,num_patches_w):
        for j in range(0,num_patches_h):
            img_reconstructed[stride*j:stride*(j+1),stride*i:stride*(i+1)]=pred_labels[count]
            count+=1
    return img_reconstructed

def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)
            loss = loss * weights 
            loss = - K.mean(loss, -1)
            return loss
        return loss

def output_prediction_FC(model, image_array, final_mask, patch_size):
    start_test = time.time()
    patch_ts = patches_with_out_overlap(image_array, patch_size, img_type = 2)  
    p_labels, probs = test_FCN(model, patch_ts)
    end_test =  time.time() - start_test
    prob_recontructed = pred_recostruction(patch_size, probs, final_mask)
    return prob_recontructed, end_test

def data_augmentation_CNN(image):
    img_rot_90 = np.rot90(image, 1)
    img_rot_180 = np.rot90(image, 2)
    horizontal_flip = np.flip(image,0)
    vertical_flip = np.flip(image,1)   
    return image, img_rot_90, img_rot_180, horizontal_flip, vertical_flip


def extract_patches_CNN(input_image, reference, mask, patch_size, stride):
    window_shape = patch_size
    px_central = patch_size//2
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    window_shape_mask = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))
    patches_mask = np.array(view_as_windows(mask, window_shape_mask, step = stride))    
    
    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    labels_c_nc = patches_ref[:,px_central,px_central]
    patches_mask = patches_mask.reshape(num_row*num_col,row,col)
    labels_tr_ts = patches_mask[:,px_central,px_central]
    return patches_array, labels_c_nc, labels_tr_ts

def get_patches_batch_CNN(image, rows, cols, radio, batch):
    temp = []
    for i in range(0, batch):
        batch_patches = image[rows[i]-radio:rows[i]+radio+1, cols[i]-radio:cols[i]+radio+1, :]
        temp.append(batch_patches)
    patches = np.asarray(temp).astype('float32')
    return patches

def get_patches_class_CNN(patches_img, labels_c_nc, labels_tr_ts_val, set_data):
  # Set data Test(0), Train(1), validation(2)
    count = 0
    count_c = 0
    count_nc = 0
    patches_c = []
    patches_ref_c = []
    patches_nc = []
    patches_ref_nc = []
    for i in range(0,len(patches_img)):    
          if labels_tr_ts_val[i]==set_data:
                if labels_c_nc[i]<2:
                      count = count+1
                      if labels_c_nc[i]==1: # train change
                            count_c = count_c+1
                            patches_c.append(patches_img[i,:,:,:])
                            patches_ref_c.append(labels_c_nc[i])
                      if labels_c_nc[i]==0: # train no change
                            count_nc = count_nc+1
                            patches_nc.append(patches_img[i,:,:,:])
                            patches_ref_nc.append(labels_c_nc[i])

    patches_c = np.asarray(patches_c)
    patches_ref_c = np.asarray(patches_ref_c)
    patches_nc = np.asarray(patches_nc)
    patches_ref_nc = np.asarray(patches_ref_nc)
    return patches_c, patches_ref_c, patches_nc, patches_ref_nc, count, count_c, count_nc

def balance_data_CNN(patches_c, label_c, patches_nc, label_nc, number_class):
    temp1 = []
    for i in range(0,len(patches_c)):
        temp1.append(data_augmentation_CNN(patches_c[i]))
    patches_c1 = np.vstack(temp1)
    patches_ref_c1 = np.ones((len(patches_c1)))*label_c
    print(len(patches_c1))
    
    patches_ref_nc = np.ones((len(patches_nc)))*label_nc
    patches_nc, patches_ref_nc = shuffle(patches_nc, patches_ref_nc , random_state = 0)
    patches_c1, patches_ref_c1 = shuffle(patches_c1, patches_ref_c1 , random_state = 0)

    patches_nc1 = patches_nc[:len(patches_ref_c1),:,:,:]
    patches_ref_nc1 = patches_ref_nc[:len(patches_ref_c1)]
    patches_bal = np.concatenate((patches_c1, patches_nc1), axis=0)
    patches_bal_ref = np.concatenate((patches_ref_c1, patches_ref_nc1), axis=0)
    patches_bal, patches_bal_ref = shuffle(patches_bal, patches_bal_ref , random_state = 0)
    patches_bal_h = tf.keras.utils.to_categorical(patches_bal_ref, number_class)
    return patches_bal, patches_bal_h

