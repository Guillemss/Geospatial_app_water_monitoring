#ifndef INFERENCE_H
#define INFERENCE_H

#include "onnxruntime_c_api.h"
#include "../evaluate.h"

/* Run single inference, write output, optionally evaluate against label */
double run_inference(OrtSession* session, const ORTCHAR_T* input_file,
                     const ORTCHAR_T* output_file, const char* label_file, SegEval* eval);

/* Print model input/output tensor info */
void print_model_info(OrtSession* session);

/* Setup CUDA execution provider */
int enable_cuda(OrtSessionOptions* session_options);

#endif
