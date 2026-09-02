## Tflite-Micro models

These models are basically the same flatbuffer tflite models like in the tflite folder, just that they have a fixed input size. In tflite-micro it is not possible to resize the input size at runtime, while tflite (Python and C API) provides a runtime function to do this. If you need a different input size then (384, 384, 4) open an issue.
