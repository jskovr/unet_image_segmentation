#!/bin/bash

DATASET_ID="$1"
MODEL_CONFIG="$2" # (nnUNetPlans) the e.g. 3d_fullres (inside the nnUNetPlans.json, which houses the architecture)
TRAINER="$3" # (nnUNetTrainer) the e.g. nnUNetTrainerDA5_1000epochs (a python class, which has things like epochs, batch size, etc.)
FOLD="$7"
PLANNER="$8"

echo "Training model"
echo "$FOLD validation training."
echo "$PLANNER planner"
# nnUNet_raw=$4 nnUNet_results=$5 nnUNet_preprocessed=$6 nnUNetv2_train $DATASET_ID $MODEL_CONFIG "$FOLD" -tr $TRAINER --c
nnUNet_raw=$4 nnUNet_results=$5 nnUNet_preprocessed=$6 nnUNetv2_train $DATASET_ID $MODEL_CONFIG "$FOLD" -tr $TRAINER -p $PLANNER --c


echo "All tasks completed."
