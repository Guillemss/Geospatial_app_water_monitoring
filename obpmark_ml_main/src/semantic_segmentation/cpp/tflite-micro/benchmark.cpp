#include <cstdio>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "inference.h"
#include "model_data.h"
#include "image_data.h"
#include "label_data.h"

// Adjust for your target platform
#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (30 * 1024 * 1024)
#endif

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// UNet cloud segmentation model ops
using OpResolver = tflite::MicroMutableOpResolver<10>;

static TfLiteStatus RegisterOps(OpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
    TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddResizeBilinear());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    return kTfLiteOk;
}

static OpResolver op_resolver;

int main() {
    printf("=== OBPMark-ML Semantic Segmentation (TFLite Micro) ===\n");
    printf("Image: %dx%dx%d\n", IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS);

    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        fprintf(stderr, "Model schema %lu != expected %d\n",
                (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    if (RegisterOps(op_resolver) != kTfLiteOk) {
        fprintf(stderr, "RegisterOps() failed\n");
        return 1;
    }

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "AllocateTensors() failed\n");
        return 1;
    }
    printf("Arena used: %zu bytes\n", interpreter.arena_used_bytes());

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    print_model_summary(&interpreter);

    prepare_input(input, image_raw, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS);

    double elapsed = run_inference(&interpreter);
    if (elapsed < 0) return 1;

    printf("Inference time: %f seconds\n", elapsed);

    evaluate_output(output, label_raw, IMAGE_WIDTH, IMAGE_HEIGHT);

    return 0;
}
