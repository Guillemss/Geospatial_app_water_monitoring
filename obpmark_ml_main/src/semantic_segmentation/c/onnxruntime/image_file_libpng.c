#include <png.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "image_file.h"

int read_image_file(const ORTCHAR_T* input_file, size_t* height, size_t* width, float** out, size_t* output_count) {
  png_image image; /* The control structure used by libpng */
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, input_file) == 0) {
    return -1;
  }
  /* Force RGBA format (4 channels) for semantic segmentation input */
  image.format = PNG_FORMAT_RGBA;
  uint8_t* buffer;
  size_t input_data_length = PNG_IMAGE_SIZE(image);
  buffer = (uint8_t*)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer, 0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return -1;
  }
  hwc_to_chw(buffer, image.height, image.width, out, output_count);
  free(buffer);
  *width = image.width;
  *height = image.height;
  return 0;
}


int read_label_file(const ORTCHAR_T* label_file, size_t* height, size_t* width, uint8_t** out) {
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, label_file) == 0) {
    return -1;
  }
  image.format = PNG_FORMAT_GRAY;
  size_t data_length = PNG_IMAGE_SIZE(image);
  *out = (uint8_t*)malloc(data_length);
  if (png_image_finish_read(&image, NULL, *out, 0, NULL) == 0) {
    free(*out);
    return -1;
  }
  *width = image.width;
  *height = image.height;
  return 0;
}


int write_image_file(uint8_t* model_output_bytes, unsigned int height,
                     unsigned int width, const ORTCHAR_T* output_file){
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_GRAY;
  image.height = height;
  image.width = width;
  int ret = 0;
  if (png_image_write_to_file(&image, output_file, 0 /*convert_to_8bit*/, model_output_bytes, 0 /*row_stride*/,
			      NULL /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", output_file, image.message);
    ret = -1;
  }
  return ret;
}
