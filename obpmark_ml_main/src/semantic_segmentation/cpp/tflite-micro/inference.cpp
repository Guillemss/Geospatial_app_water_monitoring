#include "inference.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


void prepare_input(TfLiteTensor* input,
                   const uint8_t* image_raw, int width, int height, int channels) {
    // NHWC float32, normalize to 0..1
    int total = width * height * channels;
    for (int i = 0; i < total; i++) {
        input->data.f[i] = image_raw[i] / 255.0f;
    }
}


double run_inference(tflite::MicroInterpreter* interpreter) {
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        fprintf(stderr, "Invoke() failed\n");
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    return (double)(ts_end.tv_sec - ts_start.tv_sec) +
           (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
}


void print_model_summary(tflite::MicroInterpreter* interpreter) {
    printf("Input tensors:  %zu\n", interpreter->inputs_size());
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
        TfLiteTensor* t = interpreter->input(i);
        printf("  [%zu] dims=(", i);
        for (int d = 0; d < t->dims->size; d++) {
            printf("%d%s", t->dims->data[d], d < t->dims->size - 1 ? "," : "");
        }
        printf(") type=%s\n", TfLiteTypeGetName(t->type));
    }

    printf("Output tensors: %zu\n", interpreter->outputs_size());
    for (size_t i = 0; i < interpreter->outputs_size(); i++) {
        TfLiteTensor* t = interpreter->output(i);
        printf("  [%zu] dims=(", i);
        for (int d = 0; d < t->dims->size; d++) {
            printf("%d%s", t->dims->data[d], d < t->dims->size - 1 ? "," : "");
        }
        printf(") type=%s\n", TfLiteTypeGetName(t->type));
    }
}


void evaluate_output(TfLiteTensor* output,
                     const uint8_t* label_raw, int width, int height) {
    float* output_data = output->data.f;

    int total = width * height;
    int tp = 0, fp = 0, fn = 0, tn = 0;

    for (int i = 0; i < total; i++) {
        uint8_t pred = (output_data[i] > 0.5f) ? 1 : 0;
        uint8_t gt   = (label_raw[i] > 0) ? 1 : 0;

        if (pred == 1 && gt == 1) tp++;
        else if (pred == 1 && gt == 0) fp++;
        else if (pred == 0 && gt == 1) fn++;
        else tn++;
    }

    printf("\nPredicted Cloud Coverage: %f\n", (float)(tp+fp)/(width*height));

    float precision = (tp + fp > 0) ? (float)tp / (float)(tp + fp) : 0.0f;
    float recall    = (tp + fn > 0) ? (float)tp / (float)(tp + fn) : 0.0f;
    float f1        = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    printf("\n==== Semantic Segmentation Evaluation ====\n");
    printf("Pixels: %d (TP=%d, FP=%d, FN=%d, TN=%d)\n", total, tp, fp, fn, tn);
    printf("Precision: %.3f\n", precision);
    printf("Recall:    %.3f\n", recall);
    printf("F1 Score:  %.3f\n", f1);
    printf("==========================================\n");
}
