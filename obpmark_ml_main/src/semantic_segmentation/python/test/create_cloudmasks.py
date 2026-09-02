import tensorflow as tf
from semantic_segmentation.python.dataset.Dataset_tf import Cloud95DatasetTest
import numpy as np
import tifffile as tiff
import os
import csv

# change these paths appropriately
saved_model_dir = '../models/unet_32_alpha_0_25_bilinear'  # path to the tested model
test_csv = 'test_patches_38-Cloud.csv'  # name of csv holding the patch names of the test set
rootpath = '../../data/38-Cloud_test/'  # folder where the test data lies in
pred_dir = 'gt_patches'  # folder to save predicted patches

with open(rootpath + test_csv, newline='') as f:
    reader = csv.reader(f)
    data = np.array(list(reader))[1:,0]

batch_size = 12
img_size = (384, 384)

testdata_gen = Cloud95DatasetTest(data[1:], rootpath, batch_size, img_size)

# choose here your prefered backend
from semantic_segmentation.python.inference.backend_tf import BackendTensorflow as B

backend = B()
backend.load(saved_model_dir)

for x, y in testdata_gen:
    images = backend.predict(x)
    for image, image_id in zip(images, y):
        image = np.array(image[:, :, 0]).astype(np.float32)
        tiff.imsave(os.path.join(rootpath, pred_dir, str(image_id)) + '.TIF', image)

# old tensorflow implementation
#
# m = tf.keras.models.load_model(saved_model_dir)
#
# with tf.device('GPU:0'):
#     predictions = m.predict(testdata_gen, batch_size=32)
#
# for image, image_id in zip(predictions, data):
#     image = (np.array(image[:, :, 0])).astype(np.float32)
#     tiff.imsave(os.path.join(rootpath, pred_dir, str(image_id[0])) + '.TIF', image)
