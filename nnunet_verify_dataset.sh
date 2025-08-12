#!/bin/bash

# Verify dataset integrity
# echo "Verifying dataset integrity..."

DATASET_ID="$1"

nnUNet_raw="$2" nnUNet_results="$3" nnUNet_preprocessed="$4" nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

echo "All tasks completed."