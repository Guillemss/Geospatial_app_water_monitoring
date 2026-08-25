"""
Benchmark script for semantic segmentation
"""

import numpy as np
from tqdm import tqdm
from src.semantic_segmentation.data.utils import CloudDataloader

## chose runtime backend (tflite, tf, pytorch, onnx)
def import_onnx():
    from src.semantic_segmentation.python.inference.backend_onnxruntime import BackendOnnxruntime
    return BackendOnnxruntime()

def import_tf():
    from src.semantic_segmentation.python.inference.backend_tf import BackendTensorflow
    return BackendTensorflow()

def import_tflite():
    from src.semantic_segmentation.python.inference.backend_tflite import BackendTflite
    return BackendTflite()

def import_pytorch():
    from src.semantic_segmentation.python.inference.backend_pytorch_native import BackendPytorchNative
    return BackendPytorchNative()

def import_openvino():
    raise NotImplementedError("OpenVINO is not implemented.")

FRAMEWORKS = {
    'onnx': import_onnx,
    'tf': import_tf,
    'tflite': import_tflite,
    'pytorch': import_pytorch,
    'openvino': import_openvino,
}

PRECISION = {'onnx': ['fp32', 'int8'],
             'tf': ['fp32'],
             'tflite': ['fp32', 'fp16', 'int8', 'int16x8'],
             'pytorch': ['fp32'],
             'openvino': []}  # added later

def benchmarking(args):
    resolution = args.resolution

    ## initialize dataloaders
    warmup_generator = CloudDataloader("src/semantic_segmentation/data/imgs/" + resolution,
                                       "src/semantic_segmentation/data/labels/" + resolution)
    bench_generator = CloudDataloader("src/semantic_segmentation/data/imgs/" + resolution,
                                      "src/semantic_segmentation/data/labels/" + resolution)

    ## instantiate inference runtime
    model = FRAMEWORKS[args.framework]()
    print(f"Chose {model.name()} as runtime backend!")

    ## check if precision is supported by framework
    if args.precision not in PRECISION[args.framework]:
        raise RuntimeError(f"Chosen precision {args.precision} is not supported by framework {model.name()}")

    ## prepare model by loading weights
    model.load(precision=args.precision, acceleration=args.acceleration, resolution=resolution)

    ## warm up of the hardware
    i = 0
    for x, y in warmup_generator:
        x = model.preprocess(x)
        output_raw, _ = model.predict(np.array([x]))
        _, _ = model.postprocess(output_raw)

    ## run model
    t_inf = 0
    t_post = 0

    # Online accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for x, y in tqdm(bench_generator):
        x = model.preprocess(x)
        output_raw, dt_inf = model.predict(np.array([x]))
        output, dt_post = model.postprocess(output_raw)
        t_inf += dt_inf
        t_post += dt_post

        # Flatten to 1D for simplicity
        pred_flat = output.flatten()
        gt_flat = y.flatten()

        # Compute TP, FP, FN directly
        tp = np.sum((pred_flat == 1) & (gt_flat == 1))
        fp = np.sum((pred_flat == 1) & (gt_flat == 0))
        fn = np.sum((pred_flat == 0) & (gt_flat == 1))

        # Accumulate globally
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # to milliseconds
    t_inf *= 1000
    t_post *= 1000
    t_inf_avg = t_inf / len(bench_generator)
    t_post_avg = t_post / len(bench_generator)

    print(f"Total inference time for {len(bench_generator)} samples was {t_inf+t_post:.1f} ms")
    print(f"Average time for one sample {t_inf_avg+t_post_avg:.1f} ms")
    print(f"Composed of {t_inf_avg:.1f} ms inference and {t_post_avg:.1f} ms postprocessing")

    if args.print:
        # Compute final metrics
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Pretty print
        print("==== Binary Segmentation Metrics ====")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1 Score : {f1_score:.3f}")
        print("====================================")

if __name__=="__main__":
    pass

