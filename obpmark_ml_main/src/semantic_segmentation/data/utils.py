import os

import numpy as np
from PIL import Image
import tensorflow as tf
from onnxruntime.quantization import CalibrationDataReader

# from natsort import natsorted

class CloudDataloader(tf.keras.utils.Sequence):
    def __init__(self, path2imgs, path2labels):
        self.x = [os.path.join(path2imgs, f) for f in os.listdir(path2imgs) if not f.startswith('.')]
        self.y = [os.path.join(path2labels, f) for f in os.listdir(path2labels) if not f.startswith('.')]
        self.x.sort()
        self.y.sort()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.x[idx])).astype(np.float32)
        gt = np.array(Image.open(self.y[idx])).astype(np.float32)
        return img, gt

# ONNXRuntime calib reader wrapper
class CloudCalibReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.dataloader = dataloader
        self.input_name = input_name
        self.enum_data = None
        self.iter = 0

    def get_next(self):
        if self.enum_data is None:
            # create a generator from your dataloader
            self.enum_data = iter(
                {self.input_name: np.expand_dims(np.transpose(img, (2,0,1)), 0)}  # add batch dim and transpose to channel first
                for img, _ in self.dataloader
            )
        return next(self.enum_data, None)
