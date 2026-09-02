# import tensorflow as tf
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras.layers import SeparableConv2D
# from tensorflow.keras.layers import UpSampling2D
#
# def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
#     """
#     This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
#     The block returns the activation values for next layer along with a skip connection which will be used in the decoder
#     """
#
#     conv = Conv2D(n_filters,
#                   3,   # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(inputs)
#     conv = BatchNormalization()(conv, training=False)
#
#     conv = Conv2D(n_filters,
#                   3,   # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(conv)
#     conv = BatchNormalization()(conv, training=False)
#
#     if dropout_prob > 0:
#         conv = tf.keras.layers.Dropout(dropout_prob)(conv)
#
#     if max_pooling:
#         next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
#     else:
#         next_layer = conv
#
#     skip_connection = conv
#
#     return next_layer, skip_connection
#
# def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
#     """
#     Decoder Block first uses bilinear interpolation to upscale the image to a bigger size and then,
#     merges the result with skip layer results from encoder block
#     Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
#     The function returns the decoded layer output
#     """
#
#     # up = Conv2DTranspose(
#     #             n_filters,
#     #             (3,3),    # Kernel size
#     #             strides=(2,2),
#     #             padding='same')(prev_layer_input)
#     # height, width = prev_layer_input.shape[1:3]
#     up = UpSampling2D()(prev_layer_input)
#
#     merge = concatenate([up, skip_layer_input], axis=3)
#
#
#     conv = Conv2D(n_filters,
#                  3,     # Kernel size
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='HeNormal')(merge)
#     conv = BatchNormalization()(conv, training=False)
#
#     conv = Conv2D(n_filters,
#                  3,   # Kernel size
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='HeNormal')(conv)
#     conv = BatchNormalization()(conv, training=False)
#
#     return conv
#
# def UNetCompiled(input_size=(384, 384, 4), n_filters=32, n_classes=1):
#     """
#     Combine both encoder and decoder blocks according to the U-Net research paper
#     Return the model as output
#     """
#
#     inputs = Input(input_size)
#
#     cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
#     cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
#     cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
#     cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
#
#     ublock6 = DecoderMiniBlock(cblock4[0], cblock4[1],  n_filters * 8)
#     ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
#     ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
#     ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)
#
#     conv9 = Conv2D(n_filters,
#                  3,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(ublock9)
#
#     conv10 = Conv2D(n_classes, 1, padding='same', activation='sigmoid')(conv9)
#
#     # Define the model
#     model = tf.keras.Model(inputs=inputs, outputs=conv10)
#
#     return model


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
from tensorflow.keras.layers import UpSampling2D


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """

    if max_pooling:
        inputs = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs)

    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)

    conv = BatchNormalization()(conv, training=False)

    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    conv = BatchNormalization()(conv, training=False)

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # if max_pooling:
    #     next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    # else:
    #     next_layer = conv
    #
    # skip_connection = conv

    return conv


def DecoderMiniBlock(prev_layer_input, skip_layer_input, in_filters=32, out_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """

    # up = Conv2DTranspose(
    #     in_filters,
    #     (3, 3),  # Kernel size
    #     strides=(2, 2),
    #     padding='same')(prev_layer_input)
    # height, width = prev_layer_input.shape[1:3]
    # up = Resizing(height * 2, width * 2)(prev_layer_input)

    up = UpSampling2D(interpolation='bilinear')(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(in_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = BatchNormalization()(conv, training=False)

    conv = Conv2D(out_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization()(conv, training=False)

    return conv


def UNetCompiled(input_size=(None, None, 4), n_filters=32, n_classes=1):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """

    inputs = Input(input_size)

    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=False)
    cblock2 = EncoderMiniBlock(cblock1, n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2, n_filters * 4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3, n_filters * 8, dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4, n_filters * 8, dropout_prob=0.3, max_pooling=True)
    ublock = DecoderMiniBlock(cblock5, cblock4, n_filters * 8, n_filters * 8//2)
    ublock = DecoderMiniBlock(ublock, cblock3, n_filters * 4, n_filters * 4//2)
    ublock = DecoderMiniBlock(ublock, cblock2, n_filters * 2, n_filters)
    ublock = DecoderMiniBlock(ublock, cblock1, n_filters)

    # conv9 = Conv2D(n_filters,
    #                3,
    #                activation='relu',
    #                padding='same',
    #                kernel_initializer='he_normal')(ublock)

    conv10 = Conv2D(n_classes, 1, padding='same', activation='sigmoid')(ublock)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


if __name__=="__main__":
    # from semantic_segmentation.python.training.UNetBilinear import UNetCompiled
    model_path = '/Users/jannis/final_gitlab/obpmark-ml/src/semantic_segmentation/python/models/model_small/tensorflow/fp32'
    model = UNetCompiled(n_filters=4)
    model.summary()
    model_old = tf.saved_model.load(model_path)
    model_keras = tf.keras.models.load_model(model_path)
    from keras.models import Model

    from semantic_segmentation.python.training.UNetPytorch import unet
    u = unet(32, 1, 0.125)
    # transfer weights from keras model to torch model

    layers = [l for l in model_keras.layers]
    layers2 = [l for l in model.layers]


    x = layers[0].output  # input layer are the same
    for i in range(1, len(layers)):
        if layers[i].trainable:
            model.layers[i].set_weights(layers[i].get_weights())
        else:
            x = layers[i](x)

    rootpath = '../dataset/dummy_data/38-Cloud_test'  # folder where the test data lies in
    batch_size = 1
    img_size = (384, 384)
    # img_size = (192, 192)

    dummy_data = ['patch_25_2_by_5_LC08_L1TP_002054_20160520_20170324_01_T1',
                  'patch_25_2_by_5_LC08_L1TP_035034_20160120_20170224_01_T1',
                  'patch_25_2_by_5_LC08_L1TP_039034_20160320_20170224_01_T1']

    from semantic_segmentation.python.dataset.Dataset_tf import Cloud95DatasetTest
    testdata_gen = Cloud95DatasetTest(dummy_data, rootpath, batch_size, img_size)

    # get one picture from the testset
    img_y = testdata_gen[0][0][0][:, :, :4]

    import numpy as np
    import time

    t0 = time.time()
    for i in range(10):
        a = model(np.array([img_y]))
        pass
    t1 = time.time()
    total_time = t1 - t0
    print(total_time)

    t0 = time.time()
    for i in range(10):
        b = model_keras(np.array([img_y]))
        pass
    t1 = time.time()
    total_time = t1 - t0
    print(total_time)

    im = np.random.rand(768,768,4)
    t0 = time.time()
    for i in range(10):
        c = model(np.array([im]))
        pass
    t1 = time.time()
    total_time = t1 - t0
    print(total_time)
    print(np.shape(c))

    # model.save('./hannah')

    from semantic_segmentation.python.inference.backend_onnxruntime import BackendOnnxruntime
    inf_onnx = BackendOnnxruntime()
    saved_model_dir = '.'
    inf_onnx.load(saved_model_dir + '/hannah_int8.onnx')

    # int_img_y = (0.002814182545989752 * np.array(img_y) * 100000)  # + 128
    # int_img_y2 = np.array(img_y) * 255
    # int_img_y = np.round(int_img_y).astype(np.uint8)

    print("ONNX Inference time:")
    t0 = time.time()
    for i in range(10):
        out_onnx = inf_onnx.predict(np.array([img_y]))
        pass
    t1 = time.time()
    total_time = t1 - t0
    print(total_time)

    try:
        np.testing.assert_allclose(a < 0.5, b< 0.5, rtol=1e-03, atol=1e-05)
        print("Tensorflow vs. TFLite\n" + "Same inference output!\n")
    except AssertionError as e:
        print("Tensorflow vs. TFLite" + e.args[0] + '\n')
    pass

    try:
        np.testing.assert_allclose(a < 0.5, out_onnx< 0.5, rtol=1e-03, atol=1e-05)
        print("Tensorflow vs. ONNX\n" + "Same inference output!\n")
    except AssertionError as e:
        print("Tensorflow vs. TFLite" + e.args[0] + '\n')
    pass