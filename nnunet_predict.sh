#!/bin/bash

# Set the paths to the nnUNet directories
echo "Predicting..."

echo "Current working directory: $(pwd)"

INPUT_DIR="$4"
DATASET_ID="$5"
OUTPUT_DIR="$6"
MODEL_CONFIG="$7"
TRAINER="$8"
DATASET_NAME="$9"
CHECKPOINT="${10}"
FOLD="${11}"

raw="$1"
preprocessed="$2"
results="$3"


echo "nnUNET_raw ${raw}"
echo "nnUNET_preprocessed ${preprocessed}"
echo "nnUNET_results ${results}"
echo "INPUT DIR ${INPUT_DIR}"
echo "OUTPUT DIR ${OUTPUT_DIR}"
echo "USING CHECKPOINT ${CHECKPOINT}"
echo "MODEL CONFIG ${MODEL_CONFIG}"
echo "TRAINER ${TRAINER}"
echo "DATASET_ID ${DATASET_ID}"
echo "FOLD ${FOLD}"
# echo "DATASER NAME ${DATASET_NAME}"

# which nnUNetv2_predict
nnUNet_raw="$raw" nnUNet_results="$results" nnUNet_preprocessed="$preprocessed" nnUNetv2_predict -i "$INPUT_DIR" -o "$OUTPUT_DIR" -d $DATASET_ID -c "$MODEL_CONFIG" -tr "$TRAINER" -f "$FOLD" -chk "$CHECKPOINT"

echo "All tasks completed."
