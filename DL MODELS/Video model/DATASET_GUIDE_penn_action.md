# Penn Action dataset guide

## What the dataset contains

Penn Action is a human activity dataset from the University of Pennsylvania. It contains 2,326 video sequences across 15 action categories. The annotation files include each clip's action label, a train/test flag, 13 body-joint coordinates for each frame, joint visibility, bounding boxes, and video dimensions/frame count. The official dataset page and download are linked below.

## Action categories

The 15 categories used in this project are:

`baseball_pitch`, `baseball_swing`, `bench_press`, `bowl`, `clean_and_jerk`, `golf_swing`, `jump_rope`, `jumping_jacks`, `pullup`, `pushup`, `situp`, `squat`, `strum_guitar`, `tennis_forehand`, and `tennis_serve`.

## What this project uses

This model trains on the dataset's pose annotations rather than feeding video pixels into the network. The notebook reads the `.mat` annotation files under `/content/Penn_Action/labels` and uses:

- The 13 annotated joints, with x/y coordinates and visibility for each frame.
- The official `train` flag to keep the dataset's training and test videos separated.
- The action string as the classification target.

Videos have different lengths, so the notebook samples each pose sequence at 64 evenly spaced time steps. Coordinates are normalized by the video width and height. It creates 91 values at each sampled step: 26 x/y coordinates, 13 visibility values, 26 frame-to-frame coordinate changes, and 26 joint-relative bone vectors.

## Files and download

The archive is about 3 GB. The Colab notebook checks for `/content/Penn_Action/labels` first and downloads/extracts the archive only when that folder is absent. The `/content` directory is temporary: a Colab runtime reset can remove the downloaded data.

- Official dataset page: https://dreamdragon.github.io/PennAction/
- Dataset archive: https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz

## Split and evaluation

The notebook honors the dataset's official video-level train/test flag. It creates a validation subset from the official training videos and fits normalization statistics using training data only. The test partition should be used only for final evaluation, not for choosing model settings.

The previously reported test accuracy for this project was 89.79%. This is an observed run result; a new run may differ. The model uses ground-truth annotated poses, so it does not yet demonstrate automatic action recognition from arbitrary raw video. A pose-estimation system would be needed to extract matching joints from new videos.

## Source

The Penn Action project page describes the dataset, annotation fields, and download. See the official page above for dataset details and terms of use.
