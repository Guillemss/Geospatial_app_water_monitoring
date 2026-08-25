### OBPMark-ML for TFLite
 The Tensorflow Lite Runtime C API is used to process the images using the models in tflite format.

### Build Instructions
To build the application [TFlite](https://www.tensorflow.org/lite) is necessary to be installed. It is recommended to build from [source](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library) 
at the moment as there is limited support for a pre built release for most target architectures.
Another requirement is [libpng1.6](http://www.libpng.org/pub/png/libpng.html). Either use your package manager or build it from [source](https://sourceforge.net/projects/libpng/).

```
sudo apt-get install libpng-dev
```

Be sure to have the header of the built tflite runtime and libpng in your include path and the respective libs in your library path. Then call the Makefile to build the application.

```
make
```

### Prepare data
The data needs to be prepared the same way like for the python scripts.
Download the data by invoking the download script

```
cd ../../data/
./get_cloud95
```

# Run
Command to run the application:
```
./benchmark <model_path> <input_dir (with ending /)> <output_dir (with ending /)>
```

CUDA is currently not supported for TFLite, so it is not necessary to specify `cuda` on the command line. `cpu` will be used as default. TFLite supports certain so called delegates (==execution 
provider in onnxruntime). OBPMark-ML currently does not support that because of lack of the respective Hardware. Create an issue if you want support for a specific delegate.
