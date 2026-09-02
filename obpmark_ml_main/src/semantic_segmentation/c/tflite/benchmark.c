#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <png.h>

#include "config.h"
#include "inference.h"
#include "../evaluate.h"

static void usage() { printf("usage: <model_path> <input_dir/> <output_dir/> [label_dir/]\n"); }

int main(int argc, char **argv) {
    if (argc < 4) {
      usage();
      return -1;
    }
    char* model_path = argv[1];
    char* in_dir = argv[2];
    char* out_dir = argv[3];
    char* label_dir = (argc >= 5) ? argv[4] : NULL;

    // Create the model and interpreter
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    // Peek at first image to get dimensions, then resize input tensor
    {
      DIR* peek = opendir(in_dir);
      struct dirent* peek_dir;
      if (peek) {
        while ((peek_dir = readdir(peek)) != NULL) {
          if (peek_dir->d_name[0] != '.') {
            char* peek_path = malloc(strlen(in_dir) + strlen(peek_dir->d_name) + 2);
            strcpy(peek_path, in_dir);
            strcat(peek_path, peek_dir->d_name);
            png_image peek_img;
            memset(&peek_img, 0, sizeof(peek_img));
            peek_img.version = PNG_IMAGE_VERSION;
            if (png_image_begin_read_from_file(&peek_img, peek_path)) {
              int dims[] = {1, (int)peek_img.height, (int)peek_img.width, IN_CHANNELS};
              printf("Resizing input tensor to [%d, %d, %d, %d]\n", dims[0], dims[1], dims[2], dims[3]);
              TfLiteInterpreterResizeInputTensor(interpreter, 0, dims, 4);
              png_image_free(&peek_img);
            }
            free(peek_path);
            break;
          }
        }
        closedir(peek);
      }
    }

    TfLiteInterpreterAllocateTensors(interpreter);
    PrintTfLiteModelSummary(interpreter);

    // Initialize evaluation
    SegEval eval = {0, 0, 0, 0};

    // Process all images
    DIR *d;
    struct dirent *dir;
    double t_one_sample;
    double t_all_sample = 0;
    int n_imgs = 0;

    d = opendir(in_dir);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
          if (dir->d_name[0] != '.') {
            char* in_path = malloc(strlen(in_dir) + strlen(dir->d_name) + 2);
            strcpy(in_path, in_dir);
            strcat(in_path, dir->d_name);

            char* out_path = malloc(strlen(out_dir) + strlen(dir->d_name) + 2);
            strcpy(out_path, out_dir);
            strcat(out_path, dir->d_name);

            char* lbl_path = NULL;
            if (label_dir) {
              lbl_path = malloc(strlen(label_dir) + strlen(dir->d_name) + 2);
              strcpy(lbl_path, label_dir);
              strcat(lbl_path, dir->d_name);
            }

            t_one_sample = run_inference(interpreter, in_path, out_path, lbl_path, &eval);
            if (t_one_sample == -1) { return -1; }
            t_all_sample += t_one_sample;
            n_imgs++;

            free(in_path);
            free(out_path);
            free(lbl_path);
          }
          t_one_sample = t_all_sample / (double)n_imgs;
        }
        closedir(d);
    }

    printf("\nTotal time elapsed: %f seconds\n", t_all_sample);
    printf("Average time per sample: %f seconds\n", t_one_sample);

    if (label_dir) {
      seg_eval_print(&eval);
    }

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}
