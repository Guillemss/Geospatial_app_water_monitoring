#!/usr/bin/python3

import tensorflow as tf


def tf2tflite(saved_model_dir, save_model=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    if save_model:
        with open(saved_model_dir + '/modelfp32.tflite', 'wb') as f:
            f.write(tflite_model)
    return tflite_model
