#include "evaluate.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void binarize_output(const float* input, size_t h, size_t w, uint8_t** output) {
  size_t n = h * w;
  uint8_t* out = (uint8_t*)malloc(n);
  assert(out != NULL);
  for (size_t i = 0; i < n; i++) {
    out[i] = (input[i] > 0.5f) ? 255 : 0;
  }
  *output = out;
}

void seg_eval_accumulate(SegEval* eval, const uint8_t* pred, const uint8_t* gt, size_t n_pixels) {
  for (size_t i = 0; i < n_pixels; i++) {
    int p = (pred[i] > 0) ? 1 : 0;
    int g = (gt[i] > 0)   ? 1 : 0;
    if (p == 1 && g == 1) eval->tp++;
    if (p == 1 && g == 0) eval->fp++;
    if (p == 0 && g == 1) eval->fn++;
  }
  eval->n_imgs++;
}

void seg_eval_print(const SegEval* eval) {
  double precision = (double)eval->tp / (double)(eval->tp + eval->fp + 1);
  double recall    = (double)eval->tp / (double)(eval->tp + eval->fn + 1);
  double f1 = 2.0 * (precision * recall) / (precision + recall + 1e-8);
  printf("\n==== Binary Segmentation Metrics ====\n");
  printf("Images evaluated: %d\n", eval->n_imgs);
  printf("Precision: %.3f\n", precision);
  printf("Recall   : %.3f\n", recall);
  printf("F1 Score : %.3f\n", f1);
  printf("=====================================\n");
}
