#!/bin/bash
# This script downloads the cloud95 dataset prepared for obpmark-ml in 1K, 2K, 4K and its labels

set -e  # exit on error

ARCHIVE="clouds.tar"
GDRIVE_ID="1b4rHhNQf5POZeCexLlw-J6r6c8wQVlZ0"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown not found. Install with: pip install gdown"
    exit 1
fi

# Download
echo "Downloading dataset..."
gdown "$GDRIVE_ID" -O "$ARCHIVE"

# Verify archive
if [ ! -f "$ARCHIVE" ]; then
    echo "Error: Download failed, archive not found."
    exit 1
fi

# Extract
echo "Extracting..."
tar -xf "$ARCHIVE"

# Cleanup
rm "$ARCHIVE"

echo "Done. Dataset available in ./imgs/, labels in ./labels/"
