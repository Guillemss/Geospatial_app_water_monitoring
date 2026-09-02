#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "inference.h"
#include "config.h"
#include "image_file.h"
#include "../evaluate.h"

/**
 * Normalize uint8 image to float (keeping HWC layout for TFLite NHWC models)
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t total = h * w * IN_CHANNELS;
  *output_count = total;
  float* output_data = (float*)malloc(total * sizeof(float));
  assert(output_data != NULL);
  for (size_t i = 0; i < total; ++i) {
    output_data[i] = input[i] / 255.f;
  }
  *output = output_data;
}


double run_inference(TfLiteInterpreter* interpreter, const char* input_file,
                     const char* output_file, const char* label_file, SegEval* eval) {
  double ret = 0;
  size_t input_height, input_width, model_input_ele_count;
  float* model_input;

  if (read_image_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  size_t input_size = 1;
  for (int i = 0; i < input_tensor->dims->size; i++) {
    input_size *= input_tensor->dims->data[i];
  }

  TfLiteTensorCopyFromBuffer(input_tensor, model_input, input_size * sizeof(float));

  // Execute inference (wall-clock)
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  TfLiteInterpreterInvoke(interpreter);
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  ret = (double)(ts_end.tv_sec - ts_start.tv_sec) +
        (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
  printf("One sample: %f seconds\n", ret);

  // Extract output
  const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
  size_t output_size = input_height * input_width;
  float *output = (float *)malloc(sizeof(float) * output_size);
  TfLiteTensorCopyToBuffer(output_tensor, output, output_size * sizeof(float));

  // Binarize and write prediction mask
  uint8_t* output_image_data = NULL;
  binarize_output(output, input_height, input_width, &output_image_data);

  if (write_image_file(output_image_data, input_height, input_width, output_file) != 0) {
    free(output); free(model_input); free(output_image_data);
    return -1;
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

  free(output);
  free(model_input);
  free(output_image_data);
  return ret;
}


static void PrintTfLiteTensorSummary(const TfLiteTensor *tensor) {
    printf("%s (", tensor->name);
    for (int j = 0; j < tensor->dims->size; j++) {
        int dim = tensor->dims->data[j];
        if (j != tensor->dims->size-1) {
            printf("%d,", dim);
        } else {
            printf("%d) ", dim);
        }
    }
    TfLiteType t = TfLiteTensorType(tensor);
    printf("%s ", TfLiteTypeGetName(t));
    if (t == kTfLiteUInt8) {
        TfLiteQuantizationParams qparams = TfLiteTensorQuantizationParams(tensor);
        printf("[scale=%.2f, zero_point=%d]\n", qparams.scale, qparams.zero_point);
    } else {
        printf("\n");
    }
}

void PrintTfLiteModelSummary(TfLiteInterpreter *interpreter) {
    int input_tensor_count = TfLiteInterpreterGetInputTensorCount(interpreter);
    int output_tensor_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
    printf("inputs=%d, output=%d\n", input_tensor_count, output_tensor_count);
    for (int i = 0; i < input_tensor_count; i++) {
        const TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
        printf("inp_tensor[%d]: ", i);
        PrintTfLiteTensorSummary(input_tensor);
    }
    for (int i = 0; i < output_tensor_count; i++) {
        const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
        printf("out_tensor[%d]: ", i);
        PrintTfLiteTensorSummary(output_tensor);
    }
}
