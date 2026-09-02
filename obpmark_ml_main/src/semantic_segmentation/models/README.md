# Models

## General Architecture

U-Nets have become a standard approach for segmentation tasks and were also shown to be effective on cloud screening tasks. Constraints for the use in on board processing
are the enormous amount of parameters and the high computational cost, though. A good trade-off between computational complexity/memory footprint and prediction accuracy has to be found. For this the number of parameter needs to be scaled down. 
This can be achieved through a reduction of the depth and the number of filters in convolution layers. Another method is using techniques from the MobileNetv2 architecture like Depthwise Separable Convolutions and adapt them to the U-Net architecture.


![Standard UNet architecture](https://miro.medium.com/max/1400/1*x0kR2rGlTibVbu8InCNBVg.jpeg)

The models are trained on patches with 4 Channels (R/G/B/NIR) of dimension 384x384. The architecture is fully convolutional though
and can handle also different sized patches. This can obviously lower the prediction accuracy. The patches of one batch must be of the same size.
## Problems
Quantization tools of the different frameworks have different and only limited set of operations that they can provide a
quantized operation. Especially the ONNX runtime at the moment misses transposed convolutions (can be looked up [here](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/registry.py) in the variable *QLinearOpsRegistry*). 
We found that the Tensorflow resize bilinear function is supported in the [conversion](https://github.com/onnx/tensorflow-onnx/blob/f64772ce166ea2a0402524d9741b2fb71e5663df/tf2onnx/onnx_opset/nn.py#L1348) to ONNX in the [tf2onnx package](https://github.com/onnx/tensorflow-onnx).
The original UNet Paper uses bilinear upsampling instead of transposed convolution too. So there is a second line of models
now that is being investigated using this form of upsampling showing quite comparable performance. One downside is the loss of variable input size.
## Naming convention

| Modelname | First Conv #Filter | Downscaling factor #Filter | Using SeparableConv | Upsampling     |
|---------|----|--------------------------|---------------------|----------------|
| unet    | 32 |         alpha{0-1}                 | { _ /mobile}        | { _ /bilinear} |

## Frameworks and Precisions
Main developement was performed using Tensorflow 2.8.0.

1. Tensorflow: FP32
2. Tensorflow Lite: Converted from TF in FP32 and quantized to INT8, FP16 coming soon
3. ONNX: Converted from TF in FP32. Models using bilinear upsampling instead of transposed convolution quantized to INT8 using the ONNX Runtime quantization tool. This is very experimental and the correct functionality has to be tested more thoroughly. 
4. PyTorch: Because of some inconsistency in the model definition between Tensorflow and PyTorch, the PyTorch models must be retrained and will be provided soon.