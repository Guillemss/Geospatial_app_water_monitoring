#!/bin/bash
#
# Generate C headers from model, image, and label for TFLM embedding.
#
# Usage: ./convert_data.sh <model.tflite> <image.png> <label.png>
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <model.tflite> <image.png> <label.png>"
    echo ""
    echo "Example:"
    echo "  ./convert_data.sh ../../models/tflite/fp32/model.tflite \\"
    echo "                    ../../data/imgs/sample.png \\"
    echo "                    ../../data/labels/sample.png"
    exit 1
fi

MODEL="$1"
IMAGE="$2"
LABEL="$3"

# --- model_data.h ---
echo "Converting model: $MODEL"
python3 -c "
import sys

data = open('$MODEL', 'rb').read()
with open('$SCRIPT_DIR/model_data.h', 'w') as f:
    f.write('#pragma once\n')
    f.write('#include <cstdint>\n\n')
    f.write('alignas(16) const uint8_t model_data[] = {\n')
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        f.write('    ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')
    f.write('};\n')
    f.write(f'const unsigned int model_data_len = {len(data)};\n')
print(f'  Model size: {len(data)} bytes')
"

# --- image_data.h ---
echo "Converting image: $IMAGE"
python3 -c "
from PIL import Image
import numpy as np

img = np.array(Image.open('$IMAGE'))
h, w = img.shape[:2]
c = img.shape[2] if img.ndim == 3 else 1
raw = img.astype(np.uint8).tobytes()

with open('$SCRIPT_DIR/image_data.h', 'w') as f:
    f.write('#pragma once\n')
    f.write('#include <cstdint>\n\n')
    f.write(f'#define IMAGE_WIDTH  {w}\n')
    f.write(f'#define IMAGE_HEIGHT {h}\n')
    f.write(f'#define IMAGE_CHANNELS {c}\n\n')
    f.write('const uint8_t image_raw[] = {\n')
    for i in range(0, len(raw), 16):
        chunk = raw[i:i+16]
        f.write('    ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')
    f.write('};\n')
    f.write(f'const unsigned int image_raw_len = {len(raw)};\n')
print(f'  Image: {w}x{h}x{c} ({len(raw)} bytes)')
"

# --- label_data.h ---
echo "Converting label: $LABEL"
python3 -c "
from PIL import Image
import numpy as np

lbl = np.array(Image.open('$LABEL'))
if lbl.ndim == 3:
    lbl = lbl[:, :, 0]
raw = lbl.astype(np.uint8).tobytes()

with open('$SCRIPT_DIR/label_data.h', 'w') as f:
    f.write('#pragma once\n')
    f.write('#include <cstdint>\n\n')
    f.write('const uint8_t label_raw[] = {\n')
    for i in range(0, len(raw), 16):
        chunk = raw[i:i+16]
        f.write('    ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')
    f.write('};\n')
    f.write(f'const unsigned int label_raw_len = {len(raw)};\n')
print(f'  Label: {lbl.shape} ({len(raw)} bytes)')
"

echo "Done. Generated: model_data.h, image_data.h, label_data.h"
