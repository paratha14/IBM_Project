# Penn Action Pose + Motion BiLSTM

This project classifies 15 actions in the Penn Action dataset from human pose sequences. It uses annotated body joints rather than raw video pixels, so the model learns from compact pose features.

## Files

- `penn_action_pose_bilstm_optimized_final.ipynb` — open this notebook in Google Colab and run the cells from top to bottom.
- `penn_action_pose_bilstm_optimized_final.py` — Python export of the same notebook. The notebook is the easiest version to use in Colab.
- `DATASET_GUIDE_penn_action.md` — dataset overview, labels, annotations, and how this project uses the data.
- `TEAMMATE_HANDOFF_penn_action_pose_bilstm.md` — suggested continuation tasks for a teammate.

## Run in Google Colab

1. Upload the `.ipynb` file to Google Drive, then open it with Google Colaboratory. You can also use **File → Upload notebook** in Colab.
2. In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the notebook cells in order. It checks whether `/content/Penn_Action/labels` already exists before downloading the dataset. If it is missing, the notebook downloads and extracts the roughly 3 GB Penn Action archive.
4. Keep the Colab runtime open until training and evaluation finish. A runtime reset removes data stored under `/content`, so the dataset may need to be downloaded again.

## Model and features

Each video is represented by 64 sampled frames. Each frame has 91 features:

- 26 normalized joint coordinates (13 joints × x/y)
- 13 joint visibility values
- 26 coordinate changes between adjacent frames (motion)
- 26 joint-relative bone vectors

The two-layer bidirectional LSTM has 64 hidden units, dropout 0.3, and a 15-class output layer. Training uses the official Penn Action train/test split, a validation subset from the training split, train-only feature normalization, Adam, and early stopping.

## Reported result

The earlier run reported **89.79% test accuracy** and **94.84% best validation accuracy**. Training results can vary with the random split, runtime, and library versions. The notebook preserves the 91-feature setup associated with that reported result, but the exact accuracy is not guaranteed until it is rerun.

## Saved model

After training, the notebook saves a checkpoint at `/content/penn_action_pose_bilstm_91features.pkl`. In Colab, use the Files panel to download it, or add a cell such as:

```python
from google.colab import files
files.download("/content/penn_action_pose_bilstm_91features.pkl")
```

The checkpoint contains model weights and the class/feature metadata needed to recreate the model. Load checkpoints only from a trusted source.

## Suggested next steps

1. Rerun the notebook and record test accuracy, per-class precision/recall/F1, and the confusion matrix.
2. Repeat the experiment with a few fixed seeds and report mean and spread, keeping the official test split untouched for final evaluation.
3. Inspect errors among similar actions, especially tennis forehand/serve and other swinging actions.
4. Compare against simple baselines and controlled ablations (pose only; pose plus velocity; pose plus velocity and bones).
5. For real-world video prediction, build a separate preprocessing path that extracts the same 13-joint skeleton format from new videos, then test it on videos not used for training.

Avoid selecting model settings based on test-set results. Use validation data for tuning and use the test set only for final reporting.
