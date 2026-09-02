"""
pytorch backend
"""
import torch
import numpy as np
import os

from src.semantic_segmentation.python.inference.backend import Backend
from src.semantic_segmentation.python.utils.timing import measure_time
from src.semantic_segmentation.python.training.UNetPytorch import unet


class BackendPytorchNative(Backend):
    def __init__(self):
        super(BackendPytorchNative, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = None

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native"

    def image_format(self):
        return "NCHW"

    def load(self, model_path=None, inputs=None, outputs=None, precision='fp32', acceleration='cpu', resolution=None):
        model_path = os.path.join("src/semantic_segmentation/models/pytorch/", precision, "state_dict.pth") if model_path is None else model_path

        self.sess = unet(32, 1, 0.125)
        self.sess.load_state_dict(torch.load(model_path))
        self.sess.eval()

        # Fuse conv with batchnorm and relu
        for block_name in ['cblock1', 'cblock2', 'cblock3', 'cblock4', 'cblock5',
                           'ublock4', 'ublock3', 'ublock2', 'ublock1']:
            block = getattr(self.sess, block_name)
            torch.ao.quantization.fuse_modules(
                block,
                [['conv1', 'batchnorm1', 'relu1'],
                 ['conv2', 'batchnorm2', 'relu2']],
                inplace=True
            )

        # prepare the backend
        self.sess = self.sess.to(self.device)
        return self

    @measure_time
    def predict(self, feed):
        feed = torch.tensor(feed).float().to(self.device)
        with torch.no_grad():
            output = self.sess(feed)
        return output.cpu().numpy()

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
        return feed/255

    @measure_time
    def postprocess(self, feed):
        """
        Binarization of the image
        """
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
        return "NCHW"

