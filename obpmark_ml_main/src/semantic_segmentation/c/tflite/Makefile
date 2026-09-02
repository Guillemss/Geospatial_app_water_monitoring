CC=gcc
CFLAGS=-ltensorflowlite_c -lpng

SRCS=benchmark.c inference.c image_file_libpng.c ../evaluate.c

all: $(SRCS)
	$(CC) $(SRCS) -o benchmark $(CFLAGS)
