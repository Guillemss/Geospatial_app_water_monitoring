#ifndef EVALUATE_H
#define EVALUATE_H

#include <stdint.h>
#include <stddef.h>

/* Evaluation accumulators for binary segmentation */
typedef struct {
  long long tp, fp, fn;
  int n_imgs;
} SegEval;

/* Binarize float model output to uint8 mask (>0.5 -> 255, else 0) */
void binarize_output(const float* input, size_t h, size_t w, uint8_t** output);

/* Accumulate TP/FP/FN for one prediction-label pair */
void seg_eval_accumulate(SegEval* eval, const uint8_t* pred, const uint8_t* gt, size_t n_pixels);

/* Print precision, recall, F1 */
void seg_eval_print(const SegEval* eval);

#endif
