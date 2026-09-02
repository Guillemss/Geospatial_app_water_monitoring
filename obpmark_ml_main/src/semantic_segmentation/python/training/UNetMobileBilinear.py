import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Resizing

def EncoderMiniBlock_mobile(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    
    conv = SeparableConv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = SeparableConv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    conv = BatchNormalization()(conv, training=False)

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
    else:
        next_layer = conv

    skip_connection = conv
    
    return next_layer, skip_connection

def DecoderMiniBlock_mobile(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses bilinear interpolation to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    
#    up = Conv2DTranspose(
#                 n_filters,
#                 (3,3),    # Kernel size
#                 strides=(2,2),
#                 padding='same')(prev_layer_input)
    height, width = prev_layer_input.shape[1:3]
    up = Resizing(height*2, width*2)(prev_layer_input)
    
    merge = concatenate([up, skip_layer_input], axis=3)
    
    
    conv = SeparableConv2D(n_filters,
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = SeparableConv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

def UNetCompiled_mobile(input_size=(384, 384, 4), n_filters=32, n_classes=1):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    
    inputs = Input(input_size)
        
    cblock1 = EncoderMiniBlock_mobile(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock_mobile(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock_mobile(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock_mobile(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    
    ublock6 = DecoderMiniBlock_mobile(cblock4[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock_mobile(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock_mobile(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock_mobile(ublock8, cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same', activation='sigmoid')(conv9)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
