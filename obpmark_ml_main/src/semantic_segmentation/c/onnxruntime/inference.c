#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "inference.h"
#include "config.h"
#include "image_file.h"
#include "../evaluate.h"

extern const OrtApi* g_ort;

#define ORT_CHECK_STATUS(expr)                                 \
  do {                                                         \
    OrtStatus* _status = (expr);                               \
    if (_status != NULL) {                                     \
      const char* _msg = g_ort->GetErrorMessage(_status);      \
      fprintf(stderr, "ORT error: %s\n", _msg);               \
      g_ort->ReleaseStatus(_status);                           \
      return -1;                                               \
    }                                                          \
  } while (0)

/**
 * Convert input from HWC format to CHW format (with normalization /255)
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t nChannels = IN_CHANNELS;
  size_t stride = h * w;
  *output_count = stride * nChannels;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  assert(output_data != NULL);
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != nChannels; ++c) {
      output_data[c * stride + i] = input[i * nChannels + c] / 255.f;
    }
  }
  *output = output_data;
}


int enable_cuda(OrtSessionOptions* session_options) {
  OrtCUDAProviderOptions o;
  memset(&o, 0, sizeof(o));
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
  if (onnx_status != NULL) {
    const char* msg = g_ort->GetErrorMessage(onnx_status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(onnx_status);
    return -1;
  }
  return 0;
}


double run_inference(OrtSession* session, const ORTCHAR_T* input_file,
                     const ORTCHAR_T* output_file, const char* label_file, SegEval* eval) {
  double ret = 0;
  size_t input_height, input_width, model_input_ele_count;
  float* model_input;

  if (read_image_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    fprintf(stderr, "Failed to read image: %s\n", input_file);
    return -1;
  }

  char inputName[] = "input";
  char outputName[] = "output";

  OrtMemoryInfo* memory_info;
  ORT_CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, IN_CHANNELS, (int64_t)input_height, (int64_t)input_width};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  OrtValue* input_tensor = NULL;
  ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                         input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                         &input_tensor));
  int is_tensor;
  ORT_CHECK_STATUS(g_ort->IsTensor(input_tensor, &is_tensor));
  g_ort->ReleaseMemoryInfo(memory_info);

  const char* input_names[] = {inputName};
  const char* output_names[] = {outputName};
  OrtValue* output_tensor = NULL;

  // Inference (wall-clock)
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  ORT_CHECK_STATUS(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
                               &output_tensor));
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  ret = (double)(ts_end.tv_sec - ts_start.tv_sec) +
        (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
  printf("One sample: %f seconds\n", ret);

  // Get output and binarize
  ORT_CHECK_STATUS(g_ort->IsTensor(output_tensor, &is_tensor));
  float* output_tensor_data = NULL;
  ORT_CHECK_STATUS(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
  uint8_t* output_image_data = NULL;
  binarize_output(output_tensor_data, input_height, input_width, &output_image_data);

  if (write_image_file(output_image_data, input_height, input_width, output_file) != 0) {
    ret = -1;
  }

  // Evaluation against ground truth
  if (eval && label_file) {
    size_t lbl_h, lbl_w;
    uint8_t* label_data = NULL;
    if (read_label_file(label_file, &lbl_h, &lbl_w, &label_data) == 0) {
      if (lbl_h == input_height && lbl_w == input_width) {
        seg_eval_accumulate(eval, output_image_data, label_data, input_height * input_width);
      } else {
        fprintf(stderr, "Label size %zux%zu does not match prediction %zux%zu\n",
                lbl_w, lbl_h, input_width, input_height);
      }
      free(label_data);
    } else {
      fprintf(stderr, "Could not read label: %s\n", label_file);
    }
  }

  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  free(model_input);
  free(output_image_data);
  return ret;
}


void print_model_info(OrtSession* session) {
  OrtAllocator* allocator;
  g_ort->GetAllocatorWithDefaultOptions(&allocator);

  size_t num_inputs;
  g_ort->SessionGetInputCount(session, &num_inputs);
  printf("Model inputs (%zu):\n", num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    char* name;
    g_ort->SessionGetInputName(session, i, allocator, &name);
    OrtTypeInfo* type_info;
    g_ort->SessionGetInputTypeInfo(session, i, &type_info);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    size_t num_dims;
    g_ort->GetDimensionsCount(tensor_info, &num_dims);
    int64_t dims[8];
    g_ort->GetDimensions(tensor_info, dims, num_dims);
    printf("  [%zu] name=\"%s\" shape=[", i, name);
    for (size_t d = 0; d < num_dims; d++) {
      printf("%lld%s", dims[d], d < num_dims-1 ? ", " : "");
    }
    printf("]\n");
    g_ort->ReleaseTypeInfo(type_info);
    allocator->Free(allocator, name);
  }

  size_t num_outputs;
  g_ort->SessionGetOutputCount(session, &num_outputs);
  printf("Model outputs (%zu):\n", num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    char* name;
    g_ort->SessionGetOutputName(session, i, allocator, &name);
    OrtTypeInfo* type_info;
    g_ort->SessionGetOutputTypeInfo(session, i, &type_info);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    size_t num_dims;
    g_ort->GetDimensionsCount(tensor_info, &num_dims);
    int64_t dims[8];
    g_ort->GetDimensions(tensor_info, dims, num_dims);
    printf("  [%zu] name=\"%s\" shape=[", i, name);
    for (size_t d = 0; d < num_dims; d++) {
      printf("%lld%s", dims[d], d < num_dims-1 ? ", " : "");
    }
    printf("]\n");
    g_ort->ReleaseTypeInfo(type_info);
    allocator->Free(allocator, name);
  }
}
