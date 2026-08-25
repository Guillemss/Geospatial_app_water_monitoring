"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

import onnxruntime as rt
import os
import numpy as np

from src.semantic_segmentation.python.utils.timing import measure_time
from src.semantic_segmentation.python.inference.backend import Backend


class BackendOnnxruntime(Backend):
    def __init__(self):
        super(BackendOnnxruntime, self).__init__()
        self.sess = None

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        # transpose layer inculded in onnx models
        return "NCHW"

    def load(self, model_path=None, inputs=None, outputs=None, precision='fp32', acceleration='cpu', resolution="1K"):
        """Load model and find input/outputs from the model file."""
        opt = rt.SessionOptions()

        model_path = os.path.join("src/semantic_segmentation/models/onnx", precision, "model.onnx") if model_path is None else model_path

        # load ONNX Backend
        if acceleration == 'cuda':
            self.sess = rt.InferenceSession(model_path, opt, providers=['CUDAExecutionProvider'])
            print("Using Onnxruntime Cuda backend! (if not specified differently by Warning)")
        elif acceleration == 'tensorrt':
            self.sess = rt.InferenceSession(model_path, opt, providers=['TensorrtExecutionProvider'])
            print("Using Onnxruntime Tensorrt backend! (if not specified differently by Warning)")
        elif acceleration == 'migraphx':
            self.sess = rt.InferenceSession(model_path, opt, providers=['MIGraphXExecutionProvider'])
            print("Using Onnxruntime MIGraphX backend! (if not specified differently by Warning)")
        elif acceleration == 'cpu':
            self.sess = rt.InferenceSession(model_path, opt, providers=['CPUExecutionProvider'])
            print("Using Onnxruntime CPU backend!")
        elif acceleration == 'coreml':
            self.sess = rt.InferenceSession(model_path, opt, providers=['CoreMLExecutionProvider'])
            print("Using Onnxruntime CoreML backend! (if not specified differently by Warning)")

        # get input and output names
        if not inputs:
            self.inputs = [meta.name for meta in self.sess.get_inputs()]
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = [meta.name for meta in self.sess.get_outputs()]
        else:
            self.outputs = outputs
        return self

    @measure_time
    def predict(self, feed):
        """Run the prediction."""
        return self.sess.run(self.outputs, {self.inputs[0]: feed})[0]

    def preprocess(self, feed):
        """
        Check for input dimension to be correct, if not transpose input feed
        """
        pass
        channel_idx = 3 if self.input_format() == "NHWC" else 1
        channel_idx -= 1 if len(np.shape(feed)) == 3 else channel_idx  # no batch dimension
        if np.shape(feed)[channel_idx] != 4:  # transpose necessary?
            if len(np.shape(feed)) == 4:  # with batch dimension
                feed = np.transpose(feed, (0, 3, 1, 2))
            else:  # without batch dimension
                feed = np.transpose(feed, (2, 0, 1))
        return feed/255.

    @measure_time
    def postprocess(self, feed):
        """
        Binarization of the image
        """
        # check for right dimensions
        if np.shape(feed)[1] == 1:
            feed = feed[0,0]
        else:
            feed = feed[0,...,0]
        return feed > 0.5

    def input_format(self):
        """
        N: batch dimension
        H: height
        W: width
        C: channel dimension
        """
        return "NCHW"

    def output_format(self):
        """
        N: batch dimension
        H: height
        W: width
        C: channel dimension
        """
        return "NHWC"

