# Sensor Human Activity Recognition (HAR)

This project classifies six human activities from smartphone motion-sensor sequences using PyTorch. The current Colab notebook compares the original BiLSTM with a temporal CNN-BiLSTM, chooses a model using validation data, refits the selected model on the full official training set, evaluates the official test set, and saves a checkpoint.

## Files

- `har_bilstm_sensor_enhanced_colab.ipynb` — run this notebook in Google Colab.
- `TEAMMATE_HANDOFF_sensor_har.md` — results, caveats, and suggested next steps for a teammate.

## Run the notebook

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** if available.
3. Run all cells from the top. The notebook downloads the UCI HAR archive (about 58 MB) only if it is not already extracted in the runtime.
4. Download the checkpoint when training finishes. Colab files under `/content` are temporary and may be deleted when the runtime resets.

## Dataset and input

The UCI HAR dataset contains recordings from 30 volunteers performing six activities while carrying a waist-mounted smartphone. It provides accelerometer and gyroscope measurements. This model uses 9 inertial-signal channels in 128-time-step windows: body acceleration (x/y/z), body gyroscope (x/y/z), and total acceleration (x/y/z). The sampling rate is 50 Hz, so a window represents about 2.56 seconds. Dataset details are on the [UCI Machine Learning Repository page](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones).

The six labels are `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, and `LAYING`.

## Current reported run

The latest screenshots show the temporal CNN-BiLSTM selected with **96.74% validation accuracy at epoch 11**, followed by refitting on all 7,352 official training windows. The attached confusion matrix contains 2,746 correct predictions out of 2,947 test windows, which is **93.18% test accuracy**. This percentage is calculated from the matrix; reruns may differ.

Most test errors are between `SITTING` and `STANDING`: 87 sitting windows were predicted as standing, and 103 standing windows as sitting. This is the main class pair to investigate next.

## Evaluation caveat

The current validation code randomly splits windows, not participants. Since multiple windows come from each participant, a participant may appear in both the training and validation portions. This can make validation performance optimistic for unseen people. The official test partition remains separate, so the reported test matrix is the better current estimate of generalization. See the teammate handoff for the recommended subject-disjoint validation improvement.

## Saved model

The notebook saves `/content/har_bilstm_sensor_enhanced.pkl`. Its checkpoint includes weights, selected model name, channel and class ordering, normalization values, selected epoch count, validation accuracy, and test accuracy. Keep the model and preprocessing metadata together for inference.
