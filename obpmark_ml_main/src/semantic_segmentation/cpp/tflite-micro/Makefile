CXX = g++
CXXFLAGS = -std=c++17 -O2 -DTF_LITE_STATIC_MEMORY
LDFLAGS = -L. -ltensorflow-microlite -lm

SRCS = benchmark.cpp inference.cpp

all: benchmark

benchmark: $(SRCS) model_data.h image_data.h label_data.h
	$(CXX) $(CXXFLAGS) $(SRCS) -o benchmark $(LDFLAGS)

data: convert_data.sh
	./convert_data.sh ../../models/tflite/fp32/model.tflite \
	                  ../../data/imgs/sample.png \
	                  ../../data/labels/sample.png

clean:
	rm -f benchmark
