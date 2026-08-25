#!/bin/bash
#
# OBPMark-ML Semantic Segmentation - C Benchmark Runner
#
# Usage examples:
#   ./run_benchmark.sh --framework tflite --precision fp32 --resolution 384x384 --print
#   ./run_benchmark.sh --framework onnxruntime --precision int8 --acceleration cuda --print
#

set -e

# Defaults (matching Python benchmark.py)
FRAMEWORK="tflite"
PRECISION="fp32"
RESOLUTION="384x384"
ACCELERATION="cpu"
PRINT_EVAL=false

# Paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$ROOT_DIR/semantic_segmentation/data"
MODELS_DIR="$ROOT_DIR/semantic_segmentation/models"

usage() {
    echo "OBPMark-ML Semantic Segmentation (C)"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --framework    tflite|onnxruntime        (default: tflite)"
    echo "  --precision    fp32|fp16|int8|int16x8    (default: fp32)"
    echo "  --resolution   384x384|1K|2K|4K          (default: 384x384)"
    echo "  --acceleration cpu|cuda                  (default: cpu)"
    echo "  --print        print evaluation metrics"
    echo "  --help         show this help"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)    FRAMEWORK="$2";     shift 2 ;;
        --precision)    PRECISION="$2";     shift 2 ;;
        --resolution)   RESOLUTION="$2";    shift 2 ;;
        --acceleration) ACCELERATION="$2";  shift 2 ;;
        --print)        PRINT_EVAL=true;    shift ;;
        --help|-h)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Resolve paths
IMG_DIR="$DATA_DIR/imgs/$RESOLUTION/"
LABEL_DIR="$DATA_DIR/labels/$RESOLUTION/"

if [ "$FRAMEWORK" = "tflite" ]; then
    BENCHMARK="$SCRIPT_DIR/tflite/benchmark"
    MODEL="$MODELS_DIR/tflite/$PRECISION/model.tflite"

    if [ ! -f "$BENCHMARK" ]; then
        echo "Error: benchmark binary not found at $BENCHMARK"
        echo "Run 'make' in $SCRIPT_DIR/tflite/ first"
        exit 1
    fi

    # Create output dir
    OUT_DIR="$SCRIPT_DIR/tflite/out_${RESOLUTION}/"
    mkdir -p "$OUT_DIR"

    echo "=== OBPMark-ML Semantic Segmentation (C/TFLite) ==="
    echo "Model:      $MODEL"
    echo "Images:     $IMG_DIR"
    echo "Resolution: $RESOLUTION"
    echo "Precision:  $PRECISION"
    echo ""

    if [ "$PRINT_EVAL" = true ] && [ -d "$LABEL_DIR" ]; then
        "$BENCHMARK" "$MODEL" "$IMG_DIR" "$OUT_DIR" "$LABEL_DIR"
    else
        "$BENCHMARK" "$MODEL" "$IMG_DIR" "$OUT_DIR"
    fi

elif [ "$FRAMEWORK" = "onnxruntime" ]; then
    BENCHMARK="$SCRIPT_DIR/onnxruntime/benchmark"
    MODEL="$MODELS_DIR/onnx/$PRECISION/model.onnx"

    if [ ! -f "$BENCHMARK" ]; then
        echo "Error: benchmark binary not found at $BENCHMARK"
        echo "Run 'make' in $SCRIPT_DIR/onnxruntime/ first"
        exit 1
    fi

    OUT_DIR="$SCRIPT_DIR/onnxruntime/out_${RESOLUTION}/"
    mkdir -p "$OUT_DIR"

    echo "=== OBPMark-ML Semantic Segmentation (C/ONNXRuntime) ==="
    echo "Model:        $MODEL"
    echo "Images:       $IMG_DIR"
    echo "Resolution:   $RESOLUTION"
    echo "Precision:    $PRECISION"
    echo "Acceleration: $ACCELERATION"
    echo ""

    if [ "$PRINT_EVAL" = true ] && [ -d "$LABEL_DIR" ]; then
        "$BENCHMARK" "$MODEL" "$IMG_DIR" "$OUT_DIR" "$ACCELERATION" "$LABEL_DIR"
    else
        "$BENCHMARK" "$MODEL" "$IMG_DIR" "$OUT_DIR" "$ACCELERATION"
    fi

else
    echo "Error: unknown framework '$FRAMEWORK' (use tflite or onnxruntime)"
    exit 1
fi
