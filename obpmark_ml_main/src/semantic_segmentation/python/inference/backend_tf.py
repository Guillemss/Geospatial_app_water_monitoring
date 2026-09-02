"""
tensorflow backend (https://github.com/tensorflow/tensorflow)
"""

import tensorflow as tf
import numpy as np

from src.semantic_segmentation.python.inference.backend import Backend
from src.semantic_segmentation.python.utils.timing import measure_time


class BackendTensorflow(Backend):
    def __init__(self):
        super(BackendTensorflow, self).__init__()
        self.sess = None

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tensorflow"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"

    def load(self, model_path=None, inputs=None, outputs=None, precision=None, resolution='cpu', acceleration='cpu'):
        model_path = "src/semantic_segmentation/models/tensorflow/fp32" if model_path is None else model_path
        self.outputs = outputs
        self.inputs = inputs
        # self.sess = k3.models.load_model(model_path)
        self.sess = tf.saved_model.load(model_path)
        return self

    @measure_time
    def predict(self, feed):
        return self.sess(feed, training=False).numpy()

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
        Basically Decoding and NMS
        (Sometimes it is necessary to transpose the output, i.e. openvino model)
        """
        if np.shape(feed)[1] == 1:
            feed = feed[0,0] > 0.5
        else:
            feed = feed[0,...,0] > 0.5
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

