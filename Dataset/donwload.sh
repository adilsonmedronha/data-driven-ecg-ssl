#!/bin/bash

if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing..."
    pip install gdown
fi

declare -A folders
folders["Dataset/Benchmarks/WESAD_ECG_only"]="https://drive.google.com/drive/folders/1Di_5bLumvcwAlZFtcqzW9ScDh6geaNTS?usp=sharing"
folders["Dataset/Benchmarks/WESAD_ECG"]="https://drive.google.com/drive/folders/1i53oJ1NHcbvpIDldkPd_9wdTJqfH4_kn?usp=sharing"
folders["Dataset/fragment"]="https://drive.google.com/drive/folders/1tM5I2s3_iWjfWmGkwqBo8AGO2GO05jKd?usp=sharing"

BASE_OUTPUT_DIR="Dataset"
mkdir -p "$BASE_OUTPUT_DIR"

for name in "${!folders[@]}"; do
    FOLDER_URL="${folders[$name]}"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$name"
    
    echo "📂 Downloading '$name' from: $FOLDER_URL"
    echo "📁 Saving to: $OUTPUT_DIR"
    
    if gdown --folder "$FOLDER_URL" --output "$OUTPUT_DIR"; then
        echo "✅ Download complete: $OUTPUT_DIR"
    else
        echo "❌ Failed to download: $FOLDER_URL"
    fi

    echo "-----------------------------"
done