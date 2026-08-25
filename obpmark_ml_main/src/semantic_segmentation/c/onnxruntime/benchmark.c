#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "onnxruntime_c_api.h"
#include "inference.h"
#include "../evaluate.h"

const OrtApi* g_ort = NULL;

static void usage() { printf("usage: <model_path> <input_dir/> <output_dir/> [cpu|cuda] [label_dir/]\n"); }

int main(int argc, char* argv[]) {
  if (argc < 4) {
    usage();
    return -1;
  }
  char* model_path = argv[1];
  char* in_dir = argv[2];
  char* out_dir = argv[3];
  char* execution_provider = (argc >= 5) ? argv[4] : "cpu";
  char* label_dir = (argc >= 6) ? argv[5] : NULL;

  // Initialize runtime
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
    return -1;
  }

  OrtEnv* env;
  if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "benchmark", &env)) { return -1; }

  OrtSessionOptions* session_options;
  if (g_ort->CreateSessionOptions(&session_options)) { return -1; }

  if (strcmp(execution_provider, ORT_TSTR("cuda")) == 0) {
    if (enable_cuda(session_options)) {
      fprintf(stderr, "CUDA is not available\n");
    } else {
      printf("CUDA is enabled\n");
    }
  }

  OrtSession* session;
  if (g_ort->CreateSession(env, model_path, session_options, &session)) {
    fprintf(stderr, "Failed to create session\n");
    return -1;
  }

  print_model_info(session);

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

        t_one_sample = run_inference(session, in_path, out_path, lbl_path, &eval);
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

  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);

  return 0;
}
