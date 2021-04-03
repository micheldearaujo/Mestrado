from utils_2 import Activation, Dense, Conv2D, MaxPool2D, Conv2DTranspose, Dropout, concatenate, \
Input, UpSampling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add, ZeroPadding2D, DepthwiseConv2D, \
AveragePooling2D, Lambda, Model, load_model, Sequential


#%% CNN Architecture

def CNN_model2():   
    input_img = Input(shape=(rows, cols, channels))
    conv1 = Conv2D(128 , (3 , 3) , activation='relu' , padding='same')(input_img) 
    pool1 = MaxPool2D((2 , 2))(conv1)
    conv2 = Conv2D(256 , (3 , 3) , activation='relu' , padding='same')(pool1)
    pool2 = MaxPool2D((2 , 2))(conv2)
    conv3 = Conv2D(512 , (3 , 3) , activation='relu' , padding='same')(pool2)
    #pool3 = MaxPool2D((2 , 2))(conv3)
    flatten1 = Flatten()(conv3)
    drop1 = Dropout(0.2)(flatten1)
    output = Dense(2, activation = 'softmax')(drop1)
    print (output.shape)
    return Model(input_img , output)

#%% U-Net Architecture

def unet(input_shape, n_classes):
    input_img = Input(input_shape)
    f1 = 32
    conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'conv1')(input_img) 
    pool1 = MaxPool2D((2 , 2))(conv1)
  
    conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'conv2')(pool1)
    pool2 = MaxPool2D((2 , 2))(conv2)
  
    conv3 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'conv3')(pool2)
    pool3 = MaxPool2D((2 , 2))(conv3)
  
    conv4 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'conv4')(pool3)
    pool4 = MaxPool2D((2 , 2))(conv4)
  
    conv5 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name = 'conv5')(pool4)
    #drop1 = Dropout(0.5)(conv5)

    upsample1 = Conv2D(f1*8, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling1')(UpSampling2D(size = (2,2))(conv5))
    merged1 = concatenate([conv4, upsample1], name='concatenate1')
      
    upsample2 = Conv2D(f1*4, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling2')(UpSampling2D(size = (2,2))(merged1))
    merged2 = concatenate([conv3, upsample2], name='concatenate2')
      
    upsample3 = Conv2D(f1*2, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling3')(UpSampling2D(size = (2,2))(merged2))
    merged3 = concatenate([conv2, upsample3], name='concatenate3')
     
    upsample4 = Conv2D(f1, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling4')(UpSampling2D(size = (2,2))(merged3))
    merged4 = concatenate([conv1, upsample4], name='concatenate4')
        
    output = Conv2D(n_classes,(1,1), activation = 'softmax')(merged4)
    
    return Model(input_img , output) 

#%% FC Siamese Network Architecture

def build_enconder():
    input_layer = Input((128, 128, 2), name='enco_input')
    f1 = 32
    conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name='conv1')(input_layer) 
    pool1 = MaxPool2D((2 , 2))(conv1)
  
    conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name='conv2')(pool1)
    pool2 = MaxPool2D((2 , 2))(conv2)
  
    conv3 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name='conv3')(pool2)
    pool3 = MaxPool2D((2 , 2))(conv3)
  
    conv4 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name='conv4')(pool3)
    pool4 = MaxPool2D((2 , 2))(conv4)
    
    conv5 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name='conv5')(pool4)
    #drop1 = Dropout(0.5)(conv5)

    encoder_model = Model(inputs=[input_layer], outputs=[conv1, conv2, conv3, conv4, conv5], name='encoder')
    return encoder_model

def siamese_model(input_shape):
    f1 = 32
    encoder = build_enconder()
    left_input = Input(input_shape)
    right_input = Input(input_shape)
  
    [out1_1, out2_1, out3_1, out4_1, out5_1] = encoder(left_input)
    [out1_2, out2_2, out3_2, out4_2, out5_2] = encoder(right_input)
    
    feat1 = concatenate([out5_1, out5_2], name='concatenate1')
    #feat1 = concatenate([out1, out2], name='concatenate1')
    upsample1 = Conv2D(f1*8, (3 , 3), activation = 'relu', padding = 'same', name='upsampling1')(UpSampling2D(size = (2,2))(feat1))  

    feat2 = concatenate([upsample1, out4_1, out4_2], name='concatenate2')
    upsample2 = Conv2D(f1*4, (3 , 3), activation = 'relu', padding = 'same', name='upsampling2')(UpSampling2D(size = (2,2))(feat2))
    
    feat3 = concatenate([upsample2, out3_1, out3_2], name='concatenate3')
    upsample3 = Conv2D(f1*2, (3 , 3), activation = 'relu', padding = 'same', name='upsampling3')(UpSampling2D(size = (2,2))(feat3))
    
    feat4 = concatenate([upsample3, out2_1, out2_2], name='concatenate4')
    upsample4 = Conv2D(f1, (3 , 3), activation = 'relu', padding = 'same', name='upsampling4')(UpSampling2D(size = (2,2))(feat4))
    
    feat5 = concatenate([upsample4, out1_1, out1_2], name='concatenate5')
    
    out = (Conv2D(3, (1, 1), activation = 'softmax'))(feat5)

    siamese_net = Model(inputs=[left_input,right_input],outputs=out)
   
    return siamese_net

