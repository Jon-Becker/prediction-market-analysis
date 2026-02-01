#!/bin/bash
set -e

URL="https://storage.googleapis.com/prediction-market-analysis/data.tar.zst"
OUTPUT_FILE="data.tar.zst"
DATA_DIR="data"

# Check if data directory already exists
if [ -d "$DATA_DIR" ]; then
    echo "Data directory already exists, skipping download."
    exit 0
fi

# Download file using best available tool
download() {
    if command -v aria2c &> /dev/null; then
        echo "Downloading with aria2c..."
        aria2c -x 16 -s 16 -o "$OUTPUT_FILE" "$URL"
    elif command -v curl &> /dev/null; then
        echo "aria2c not found, falling back to curl..."
        curl -L -o "$OUTPUT_FILE" "$URL"
    elif command -v wget &> /dev/null; then
        echo "aria2c and curl not found, falling back to wget..."
        wget -O "$OUTPUT_FILE" "$URL"
    else
        echo "Error: No download tool available (aria2c, curl, or wget required)."
        exit 1
    fi
}

# Extract the archive
extract() {
    if ! command -v zstd &> /dev/null; then
        echo "Error: zstd is required but not installed."
        echo "Run 'make setup' or install zstd manually."
        exit 1
    fi

    echo "Extracting $OUTPUT_FILE..."
    zstd -d "$OUTPUT_FILE" --stdout | tar -xf -
    echo "Extraction complete."
}

# Cleanup downloaded archive
cleanup() {
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Cleaning up..."
        rm "$OUTPUT_FILE"
    fi
}

# Main
download
extract
cleanup

echo "Data directory ready."
