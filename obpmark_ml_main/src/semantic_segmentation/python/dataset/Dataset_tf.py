from tensorflow import keras
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import os


class Cloud95DatasetTrain(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, img_paths, rootpath, batch_size, img_size=(384, 384)):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = [self.path_dict_train(rootpath, patch) for patch in img_paths]

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (4,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.stack([np.array(Image.open(path['red'])),
                            np.array(Image.open(path['green'])),
                            np.array(Image.open(path['blue'])),
                            np.array(Image.open(path['nir']))], axis=2)

            img = (img / np.iinfo(img.dtype).max)  # normalize
            img = resize(img, self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            mask = np.array(Image.open(path['gt']))
            mask = resize(mask, self.img_size)
            y[j, :, :, 0] = mask/255
        return x, y

    def __call__(self):
        '''
        Needed as representative dataset for the quantization procedure of tflite
        '''
        # change range of i to speed up the conversion
        for i in tqdm(range(self.__len__())):
            # get batch of images
            x, y = self.__getitem__(i)
            for j in range(self.batch_size):
                # yield always one of these image to refine
                yield [np.array([x[j]])]

    def path_dict_train(self, rootpath, patch):
        return {'red': os.getcwd() + "/" + rootpath + "/train_red_additional_to38cloud/red_" + patch + ".TIF",
                'green': rootpath + "/train_green_additional_to38cloud/green_" + patch + ".TIF",
                'blue': rootpath + "/train_blue_additional_to38cloud/blue_" + patch + ".TIF",
                'nir': rootpath + "/train_nir_additional_to38cloud/nir_" + patch + ".TIF",
                'gt': rootpath + "/train_gt_additional_to38cloud/gt_" + patch + ".TIF"}


class Cloud95DatasetTest(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, img_paths, rootpath, batch_size, img_size=(384, 384)):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = [self.path_dict_test(rootpath, patch) for patch in img_paths]
        self.patch_names = img_paths

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        print(idx)
        i = idx * self.batch_size
        batch_input_img_paths = self.img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (4,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.stack([np.array(Image.open(path['red'])),
                            np.array(Image.open(path['green'])),
                            np.array(Image.open(path['blue'])),
                            np.array(Image.open(path['nir']))], axis=2)

            img = (img / np.iinfo(img.dtype).max)  # normalize
            img = resize(img, self.img_size)
            x[j] = img
        # as there are no labels for the testset, give back name/id of patch
        y = self.patch_names[i : i + self.batch_size]
        return x, y

    def __call__(self):
        '''
        Needed as representative dataset for the quantization procedure of tflite
        '''
        # change range of i to speed up the conversion
        for i in tqdm(range(self.__len__())):
            # get batch of images
            x, y = self.__getitem__(i)
            for j in range(self.batch_size):
                # yield always one of these image to refine
                yield [np.array([x[j]])]

    def path_dict_test(self, rootpath, patch):
        return {'red': rootpath + "/test_red/red_" + patch + ".TIF",
                'green': rootpath + "/test_green/green_" + patch + ".TIF",
                'blue': rootpath + "/test_blue/blue_" + patch + ".TIF",
                'nir': rootpath + "/test_nir/nir_" + patch + ".TIF",
                'gt': rootpath + "/test_gt/gt_" + patch + ".TIF"}