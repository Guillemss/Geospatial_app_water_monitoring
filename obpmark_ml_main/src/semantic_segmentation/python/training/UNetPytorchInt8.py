import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_segmentation.python.dataset.Dataset_tf import Cloud95DatasetTest


class EncoderMiniBlock(nn.Module):
    def __init__(self, in_filters, n_filters=32, dropout_prob=0.3, max_pooling=True):
        super(EncoderMiniBlock, self).__init__()

        self.max_pool = None
        if max_pooling:
            self.max_pool = nn.MaxPool2d(2)

        self.conv1=nn.Conv2d(in_filters, n_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.batchnorm1=nn.BatchNorm2d(n_filters, momentum=0.99, eps=0.001)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.batchnorm2 = nn.BatchNorm2d(n_filters, momentum=0.99, eps=0.001)

        self.drop = None
        if dropout_prob > 0:
            self.drop = nn.Dropout(dropout_prob)


    def forward(self, x):
        if self.max_pool:
            x=self.max_pool(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        # print(x.detach().numpy()[0, 0, 10,: 10])

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.batchnorm2(x)
        if self.drop:
            x = self.drop(x)
        return x

class DecoderMiniBlock(nn.Module):
    def __init__(self, in_filters=32, n_filters=32):
        super(DecoderMiniBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_filters, in_filters // 2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(in_filters // 2, momentum=0.99, eps=0.001)

        self.conv2 = nn.Conv2d(in_filters // 2, n_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.batchnorm2 = nn.BatchNorm2d(n_filters, momentum=0.99, eps=0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.batchnorm2(x)
        return x

class unet(nn.Module):
    def __init__(self, n_channels, n_classes, a=1):
        super(unet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        n_filters = int(n_channels * a)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cblock1 = EncoderMiniBlock(4, n_filters, dropout_prob=0, max_pooling=False)
        self.cblock2 = EncoderMiniBlock(n_filters, n_filters * 2, dropout_prob=0, max_pooling=True)
        self.cblock3 = EncoderMiniBlock(n_filters * 2, n_filters * 4, dropout_prob=0, max_pooling=True)
        self.cblock4 = EncoderMiniBlock(n_filters * 4, n_filters * 8, dropout_prob=0.3, max_pooling=True)
        self.cblock5 = EncoderMiniBlock(n_filters * 8, n_filters * 8, dropout_prob=0.3, max_pooling=True)
        self.ublock4 = DecoderMiniBlock(n_filters * 8 * 2, n_filters * 8 // 2)
        self.ublock3 = DecoderMiniBlock(n_filters * 4 * 2, n_filters * 4 // 2)
        self.ublock2 = DecoderMiniBlock(n_filters * 2 * 2, n_filters)
        self.ublock1 = DecoderMiniBlock(n_filters * 2)

        self.out_conv = nn.Conv2d(n_filters * 8, n_classes, kernel_size=1)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.quant(x)
        c1 = self.cblock1(x)
        c2 = self.cblock2(c1)
        c3 = self.cblock3(c2)
        c4 = self.cblock4(c3)
        c5 = self.cblock5(c4)

        # upsample previous, then concatenate
        c5 = self.up(c5)
        merge = torch.cat([c5, c4], dim=1)
        d = self.ublock4(merge)

        d = self.up(d)
        merge = torch.cat([d, c3], dim=1)
        d = self.ublock3(merge)

        d = self.up(d)
        merge = torch.cat([d, c2], dim=1)
        d = self.ublock2(merge)

        d = self.up(d)
        merge = torch.cat([d, c1], dim=1)
        d = self.ublock1(merge)

        return self.dequant(self.sig(self.out_conv(d)))



if __name__=="__main__":
    import tensorflow as tf
    from semantic_segmentation.python.training.UNetUpsample import UNetCompiled
    model_path = '/Users/jannis/final_gitlab/obpmark-ml/src/semantic_segmentation/python/models/model_small/tensorflow/fp32'
    model_path = '/Users/jannis/Uni/obpmark-ml/semantic_segmentation/python/training/hannah'

    model_keras = tf.keras.models.load_model(model_path)
    model = unet(32, 1, 0.125)

    # transfer weights from keras model to torch model
    layers = [l for l in model_keras.layers]
    x = layers[0].output  # input layer are the same

    convs = {"conv2d": model.cblock1.conv1,
     "conv2d_1": model.cblock1.conv2,
     "conv2d_2": model.cblock2.conv1,
     "conv2d_3": model.cblock2.conv2,
     "conv2d_4": model.cblock3.conv1,
     "conv2d_5": model.cblock3.conv2,
     "conv2d_6": model.cblock4.conv1,
     "conv2d_7": model.cblock4.conv2,
     "conv2d_8": model.cblock5.conv1,
     "conv2d_9": model.cblock5.conv2,
     "conv2d_10": model.ublock4.conv1,
     "conv2d_11": model.ublock4.conv2,
     "conv2d_12": model.ublock3.conv1,
     "conv2d_13": model.ublock3.conv2,
     "conv2d_14": model.ublock2.conv1,
     "conv2d_15": model.ublock2.conv2,
     "conv2d_16": model.ublock1.conv1,
     "conv2d_17": model.ublock1.conv2,
     "conv2d_18": model.out_conv}

    batchnorms = {"batch_normalization": model.cblock1.batchnorm1,
     "batch_normalization_1": model.cblock1.batchnorm2,
     "batch_normalization_2": model.cblock2.batchnorm1,
     "batch_normalization_3": model.cblock2.batchnorm2,
     "batch_normalization_4": model.cblock3.batchnorm1,
     "batch_normalization_5": model.cblock3.batchnorm2,
     "batch_normalization_6": model.cblock4.batchnorm1,
     "batch_normalization_7": model.cblock4.batchnorm2,
     "batch_normalization_8": model.cblock5.batchnorm1,
     "batch_normalization_9": model.cblock5.batchnorm2,
     "batch_normalization_10": model.ublock4.batchnorm1,
     "batch_normalization_11": model.ublock4.batchnorm2,
     "batch_normalization_12": model.ublock3.batchnorm1,
     "batch_normalization_13": model.ublock3.batchnorm2,
     "batch_normalization_14": model.ublock2.batchnorm1,
     "batch_normalization_15": model.ublock2.batchnorm2,
     "batch_normalization_16": model.ublock1.batchnorm1,
     "batch_normalization_17": model.ublock1.batchnorm2}

    convs_keys = convs.keys()
    batchnorms_keys = batchnorms.keys()

    for i in range(1, len(layers)):
        if layers[i].name in convs_keys:
            w, b = model_keras.layers[i].get_weights()
            w = np.transpose(w, (3,2,0,1))
            convs[layers[i].name].weight = torch.nn.Parameter(torch.Tensor(w))
            convs[layers[i].name].bias = torch.nn.Parameter(torch.Tensor(b))
        if layers[i].name in batchnorms_keys:
            gamma, beta, mean, var = model_keras.layers[i].get_weights()
            batchnorms[layers[i].name].weight = torch.nn.Parameter(torch.Tensor(gamma))
            batchnorms[layers[i].name].bias = torch.nn.Parameter(torch.Tensor(beta))
            batchnorms[layers[i].name].running_mean = torch.nn.Parameter(torch.Tensor(mean), requires_grad=False)
            batchnorms[layers[i].name].running_var = torch.nn.Parameter(torch.Tensor(var), requires_grad=False)
    torch.backends.quantized.engine = 'qnnpack'
    # create a model instance
    model_fp32 = model

    # model must be set to eval mode for static quantization logic to work
    model_fp32.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model_fp32)

    from PIL import Image

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    # input_fp32 = torch.randn(4, 1, 4, 4)
    img_y = np.array(Image.open('/Users/jannis/Downloads/clouds-2/1K/scene_3_1K.png'))

    print(np.array([img_y]).shape)

    input_fp32 = np.array([img_y]) / 255.
    input_fp32 = np.transpose(input_fp32, (0,3,1,2))
    ten = torch.Tensor(input_fp32)
    model_fp32_prepared(ten)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)

    # run the model, relevant calculations will happen in int8
    res = model_int8(ten)

    rootpath = '../dataset/dummy_data/38-Cloud_test'  # folder where the test data lies in
    batch_size = 1
    img_size = (384, 384)
    dummy_data = ['patch_25_2_by_5_LC08_L1TP_002054_20160520_20170324_01_T1',
                  'patch_25_2_by_5_LC08_L1TP_035034_20160120_20170224_01_T1',
                  'patch_25_2_by_5_LC08_L1TP_039034_20160320_20170224_01_T1']

    testdata_gen = Cloud95DatasetTest(dummy_data, rootpath, batch_size, img_size)
    #
    img_y = testdata_gen[0][0][0][:, :, :4]

    from PIL import Image
    img_y = np.array(Image.open('/Users/jannis/Downloads/clouds-2/1K/scene_3_1K.png'))

    print(np.array([img_y]).shape)

    test = np.array([img_y])/255.

    # test = np.random.rand(1,384,384,4)
    test2 = np.transpose(test, (0,3,1,2))
    output = model_keras.predict(test)

    from tensorflow.keras import Model
    layer_name = 'batch_normalization'
    intermediate_layer_model = Model(inputs=model_keras.input,
                                     outputs=model_keras.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test)
    print(np.transpose(intermediate_output, (0, 3, 1, 2))[0, 0, 10, :10])

    # model.half()
    model.eval()
    output2 = model(torch.FloatTensor(test2))
    #array(0.45259163, dtype=float32)

    output = output[0,:,:,0]
    output2 = output2[0,0,:,:]
    res = res[0,0,:,:]


    try:
        np.testing.assert_allclose(output < 0.5, output2< 0.5, rtol=1e-03, atol=1e-05)
        print("Tensorflow vs. Pytorch\n" + "Same inference output!\n")
    except AssertionError as e:
        print("Tensorflow vs. TFLite" + e.args[0] + '\n')
    pass

    try:
        np.testing.assert_allclose(output < 0.5, res< 0.5, rtol=1e-03, atol=1e-05)
        print("Tensorflow vs. Pytorch Quant\n" + "Same inference output!\n")
    except AssertionError as e:
        print("Tensorflow vs. TFLite" + e.args[0] + '\n')
    pass

    print((output2<0.5)==(output<0.5))

    import matplotlib.pyplot as plt
    plt.imshow(output2.detach().numpy()>0.5)
    plt.show()
    plt.imshow(output>0.5)
    plt.show()

    torch.save(model_int8, 'model_int8.pth')
    torch.save(model_int8.state_dict(), 'state_dict_int8.pth')
