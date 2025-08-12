#!/bin/bash

# Verify dataset integrity
# echo "Verifying dataset integrity..."

DATASET_ID="$1"
PLANNER="$5"

nnUNet_raw="$2" nnUNet_results="$3" nnUNet_preprocessed="$4" nnUNetv2_plan_experiment -d "$DATASET_ID" -pl "$PLANNER"

echo "All tasks completed."