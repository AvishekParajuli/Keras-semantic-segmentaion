import keras as k

def conv_block(ipTensor, nfilters, size=3):
    '''Convolutional block for a U_net
    Architecture: Conv2d->BN->Relu->Conv2d->BN->Relu'''
    x = k.layers.Conv2D(nfilters,kernel_size=size, padding='same')(ipTensor)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Conv2D(nfilters,kernel_size=size, padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    return x

def deconv_block(ipTensor, skipTensor,nfilters, size=3, strides=(2,2)):
    '''De-Convolutional block for a U_net
        Architecture: Conv2dTrans->Concatenate->Conv2d->BN->Relu'''
    y = k.layers.Conv2DTranspose(nfilters,kernel_size=size, padding='same', strides=strides)(ipTensor)
    y = k.layers.concatenate([y, skipTensor], axis=3)
    y = conv_block(y, nfilters,size)
    return y

def Unet(img_height, img_width, num_outclasses= 3, n_img_channels=3, \
         initial_nfilters=64,pretrained_weights =None, dropRate=0.25):
    '''Vanilla Unet architecture based model with 4-skip connection
    '''

    inputs = k.layers.Input(shape=(img_height,img_width,n_img_channels), name="input_image")
    # begin downscaling
    conv1 = conv_block(inputs, nfilters=initial_nfilters)#64
    conv1_out = k.layers.MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = conv_block(conv1_out, nfilters=2*initial_nfilters)#128
    conv2_out = k.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(conv2_out, nfilters=4 * initial_nfilters)#256
    conv3_out = k.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(conv3_out, nfilters=8 * initial_nfilters)# 512
    conv4_out = k.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = k.layers.Dropout(rate = dropRate)(conv4_out)

    conv5 = conv_block(conv4_out, nfilters=16 * initial_nfilters)  # 1024
    conv5_out = k.layers.Dropout(rate = dropRate)(conv5)

    #begin upscaling
    deconv4 = deconv_block(conv5_out, skipTensor=conv4_out, nfilters=8 * initial_nfilters)
    deconv4 = k.layers.Dropout(rate=dropRate)(deconv4)
    deconv3 = deconv_block(deconv4, skipTensor=conv3_out, nfilters=4 * initial_nfilters)
    deconv3 = k.layers.Dropout(rate=dropRate)(deconv3)
    deconv2 = deconv_block(deconv3, skipTensor=conv2_out, nfilters=2 * initial_nfilters)
    deconv1 = deconv_block(deconv2, skipTensor=conv1_out, nfilters=1 * initial_nfilters)

    #output
    output = k.layers.Conv2D(filters=num_outclasses,padding='same', kernel_size=1)(deconv1)
    output = k.layers.BatchNormalization()(output)
    output = k.layers.Activation("softmax")(output)

    model = k.Model(inputs, output, name ="Unet")
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


