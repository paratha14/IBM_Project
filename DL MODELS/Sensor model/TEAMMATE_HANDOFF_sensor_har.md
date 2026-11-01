# Teammate handoff: sensor HAR model

## Project goal

Classify six activities from short smartphone inertial-sensor windows. The current model uses nine accelerometer/gyroscope signal channels and compares a reference BiLSTM with a temporal CNN-BiLSTM.

## Start here

Run `har_bilstm_sensor_enhanced_colab.ipynb` in Google Colab, from top to bottom. Choose a GPU runtime if available. The notebook selects an architecture and epoch count using validation accuracy, refits that selection on all official training examples, evaluates the official test set once, and downloads a checkpoint.

## Latest result shared by the user

- Validation selection: Temporal CNN-BiLSTM, **96.74% validation accuracy at epoch 11**.
- Refit: selected model retrained for 11 epochs on all **7,352 official training windows**.
- Test confusion matrix: **2,746 / 2,947 correct = 93.18% accuracy** (computed from the shared matrix).
- Earlier run: **84.66% test accuracy**. The matrix above appears to be a later, improved run; keep the actual Colab output with the report when recording final results.

The largest visible error is confusion between sitting and standing: 87 sitting examples were predicted as standing, and 103 standing examples were predicted as sitting. The other activity pairs in the matrix are mostly classified correctly.

## Evaluation caveat to fix next

The notebook's validation split uses a stratified random split of windows. The UCI recordings contain multiple windows per person, so windows from the same person can occur in both training and validation. This can inflate validation accuracy when the goal is generalization to unseen people.

The official train/test partition is participant-based, and the final test set stays separate in the notebook. Keep that test set untouched. For the next version, use the training subject IDs (`subject_train.txt`) to create a subject-disjoint validation split within the official training partition. Then rerun model and epoch selection using that validation split. This may lower validation accuracy while providing a more realistic estimate for new participants. The UCI dataset description documents its volunteer-based partitioning: https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.

## Recommended next tasks

1. Add subject-disjoint validation using participant IDs; leave the official test set untouched.
2. Rerun the reference BiLSTM and CNN-BiLSTM with the same grouped split and report the best validation score and selected epoch.
3. Evaluate the selected model once on the official test set and save the accuracy, classification report, confusion matrix, seed, and library versions.
4. Inspect sitting-versus-standing errors. Compare their raw sensor traces and test small, documented feature or architecture changes using validation data only.
5. Repeat with a few fixed seeds and report the mean and spread rather than relying on one run.
6. Test inference on separately collected sensor windows only after confirming the same sampling rate, window length, channel order, and normalization.

## Reproducibility and limitations

- Input shape: 128 time steps × 9 channels.
- Label IDs are zero-based in the model; preserve `activity_names` and channel order from the checkpoint.
- The current result depends on annotated, pre-windowed UCI signals. It does not establish performance on arbitrary phone placements or new devices.
- Colab's `/content` storage is temporary. Download the checkpoint and results before ending the runtime.
- Do not select hyperparameters using the official test score.
