#ifndef INFERENCE_H
#define INFERENCE_H

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "../evaluate.h"

/* Run single inference, write output, optionally evaluate against label */
double run_inference(TfLiteInterpreter* interpreter, const char* input_file,
                     const char* output_file, const char* label_file, SegEval* eval);

/* Print model input/output tensor info */
void PrintTfLiteModelSummary(TfLiteInterpreter *interpreter);

#endif
