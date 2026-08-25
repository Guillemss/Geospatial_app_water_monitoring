#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/c/common.h"

// Prepare input tensor: RGBA uint8 -> NHWC float32, normalized /255
void prepare_input(TfLiteTensor* input,
                   const uint8_t* image_raw, int width, int height, int channels);

// Run inference and return wall-clock time in seconds
double run_inference(tflite::MicroInterpreter* interpreter);

// Print input/output tensor info
void print_model_summary(tflite::MicroInterpreter* interpreter);

// Binarize float output, evaluate against ground truth label, print metrics
void evaluate_output(TfLiteTensor* output,
                     const uint8_t* label_raw, int width, int height);
