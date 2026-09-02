#!/usr/bin/python3
import tensorflow as tf


def quantize2int8(saved_model_dir, gen, save=False):
    # gen: use here the dataset class from dataset_tf as representative dataset

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()

    if save:
        with open(saved_model_dir + '/modelint8.tflite', 'wb') as f:
            f.write(tflite_quant_model)

    return tflite_quant_model
