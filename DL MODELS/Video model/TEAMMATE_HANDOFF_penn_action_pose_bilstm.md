# Teammate handoff: Penn Action pose BiLSTM

## Goal

Continue the human activity recognition project by reproducing and evaluating a pose-based action classifier on Penn Action. The existing model uses annotated pose sequences and a bidirectional LSTM.

## Start here

Open `penn_action_pose_bilstm_optimized_final.ipynb` in Google Colab. Select a GPU runtime if one is available, then run the notebook from the first cell to the last. The notebook checks for the extracted dataset before downloading it. If the runtime was reset, `/content` may be empty and the approximately 3 GB archive will need to be downloaded again.

The paired `.py` file is an export for reading or editing outside Colab. Use the notebook for the guided run.

## Current implementation

- Dataset: Penn Action, 15 action classes; its official training/test designation is used.
- Input: 64 time steps per clip and 91 pose-derived features per frame.
- Features: normalized x/y joint positions, visibility, frame-to-frame coordinate changes, and joint-relative bone vectors.
- Model: two-layer bidirectional LSTM, 64 hidden units, 0.3 dropout, 15 output classes.
- Training: validation split from the official training partition, normalization fitted on training data only, Adam optimizer, early stopping.
- Checkpoint: `/content/penn_action_pose_bilstm_91features.pkl`.

## Previously reported result

The prior Colab run reported 89.79% test accuracy and 94.84% best validation accuracy. These are prior observed numbers, not a guarantee for a fresh run. Keep the official test partition out of model selection; tuning against it can make the final score misleading.

## Recommended work

1. Reproduce the run and save the complete output, including the classification report and confusion matrix.
2. Confirm class ordering and feature dimension (91) in the saved checkpoint and inference code.
3. Evaluate repeatability across several seeds and summarize mean and standard deviation.
4. Run controlled feature ablations using the same split and training protocol.
5. Review per-class errors, focusing on tennis forehand versus tennis serve and other visually similar motions.
6. Add an inference example for a new video only after implementing a pose extractor that returns the same 13-joint layout and feature ordering.

## Important reproducibility notes

- Record the Colab date, GPU type, PyTorch/NumPy/scikit-learn versions, seed, and best validation epoch.
- A Colab runtime reset deletes files under `/content`; save the checkpoint and any results to Drive before disconnecting.
- Do not compare runs if the data split, preprocessing, or feature ordering changed without documenting it.
- Avoid claiming that the model recognizes arbitrary raw videos until a matching pose-extraction pipeline has been tested.
