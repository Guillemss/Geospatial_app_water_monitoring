### OBPMark-ML for Onnxruntime
 The ONNX Runtime C API is used to process the images using the models in ONNX format.

### Build Instructions
To build the application the [onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.13.1) is necessary to be installed. It can either be build from source or download a pre built release for the respective target architecture.
Another requirement is [libpng1.6](http://www.libpng.org/pub/png/libpng.html). Either use your package manager or build it from [source](https://sourceforge.net/projects/libpng/).

```
sudo apt-get install libpng-dev
```

Be sure to have the header of the onnxruntime and libpng in your include path and the respective libs in your library path. Then call the Makefile to build the application.

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
./benchmark <model_path> <input_dir (with ending /)> <output_dir (with ending /)> [cpu|cuda] (cuda needs the proper onnxruntime)
```

To use the CUDA execution provider, specify `cuda` on the command line. `cpu` is the default.
