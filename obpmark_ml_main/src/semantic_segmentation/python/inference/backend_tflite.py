"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

import numpy as np
import os

from src.semantic_segmentation.python.utils.timing import measure_time
from src.semantic_segmentation.python.inference.backend import Backend

try:
    # try dedicated tflite package first
    import tflite_runtime
    import tflite_runtime.interpreter as tflite
    _version = tflite_runtime.__version__
    _git_version = tflite_runtime.__git_version__
except:
    # fall back to tflite bundled in tensorflow
    import tensorflow as tf
    from tensorflow.lite.python import interpreter as tflite
    _version = tf.__version__
    _git_version = tf.__git_version__


class BackendTflite(Backend):
    def __init__(self):
        super(BackendTflite, self).__init__()
        self.sess = None
        self.reso_dict = {"384x384": (1, 384, 384, 4),
                          "1K": (1, 1024, 1024, 4),
                          "2K": (1, 2048, 2048, 4),
                          "4K": (1, 4096, 4096, 4), }

    def version(self):
        return _version + "/" + _git_version

    def name(self):
        return "tflite"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path=None, inputs=None, outputs=None, precision='fp32', acceleration='cpu', resolution="1K"):
        model_path = os.path.join("src/semantic_segmentation/models/tflite", precision, "model.tflite") if model_path is None else model_path

        self.sess = tflite.Interpreter(model_path=model_path, num_threads=1)
        self.sess.resize_tensor_input(self.sess.get_input_details()[0]['index'], self.reso_dict[resolution])

        self.sess.allocate_tensors()
        # keep input/output details
        self.inputs = self.sess.get_input_details()[0]['index']
        self.outputs = self.sess.get_output_details()[0]['index']
        return self

    @measure_time
    def predict(self, feed):
        self.sess.set_tensor(self.inputs, feed)
        self.sess.invoke()
        # get results
        return self.sess.get_tensor(self.outputs)

    def preprocess(self, feed):
        """
        Check for input dimension to be correct, if not transpose input feed
        """
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
        if np.shape(feed)[1] == 1:
            feed = feed[0, 0]
        else:
            feed = feed[0, ..., 0]
        return feed > 0.5

    def input_format(self):
        """
        N: batch dimension
        H: height
        W: width
        C: channel dimension
        """
        return "NHWC"

    def output_format(self):
        """
        N: batch dimension
        H: height
        W: width
        C: channel dimension
        """
        return "NHWC"

