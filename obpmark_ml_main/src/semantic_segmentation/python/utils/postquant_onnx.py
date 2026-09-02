#!/usr/bin/python3

import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat


def quantize2int_onnx(saved_model_dir, gen):
    # gen: use here the dataset class from dataset_tf as representative dataset
    onnx_path = saved_model_dir + '/modelfp32.onnx'

    class CloudDataReader(CalibrationDataReader):
        '''
        Kind of the same like the representative dataset in tflite
        '''
        def __init__(self, gen):
            self.enum_data_dicts = []
            self.datasize = 0
            self.gen = gen
            self.nhwc_data_list = self.gen[0][0]
            self.preprocess_flag = True

        def get_next(self):
            if self.preprocess_flag:
                self.preprocess_flag = False
                self.datasize = len(self.nhwc_data_list)
                self.enum_data_dicts = iter([{'serving_default_input_11:0': np.array([nhwc_data])} for nhwc_data in self.nhwc_data_list])
            return next(self.enum_data_dicts, None)

    dr = CloudDataReader(gen)
    quantize_static(onnx_path, saved_model_dir+'/modelint8.onnx',  calibration_data_reader=dr, quant_format=QuantFormat.QOperator)
