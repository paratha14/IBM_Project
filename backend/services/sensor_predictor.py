"""
services/sensor_predictor.py
-----------------------------
Runs inference with the sensor HAR model and returns a structured result.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from backend.models.loader import SensorModelBundle
from backend.services.sensor_preprocessor import preprocess_sensor_window

import numpy as np


@dataclass
class SensorPrediction:
    activity: str               # predicted class label
    confidence: float           # softmax probability of the predicted class
    all_scores: dict[str, float]  # softmax probabilities for every class
    inference_ms: float         # wall-clock inference time in milliseconds


def predict_from_window(
    total_acc: np.ndarray,
    gyro: np.ndarray,
    bundle: SensorModelBundle,
) -> SensorPrediction:
    """
    Run a single sensor window through the HAR model.

    Args:
        total_acc: (128, 3) float32 array — total acceleration [x, y, z]
        gyro:      (128, 3) float32 array — gyroscope [x, y, z]
        bundle:    loaded SensorModelBundle

    Returns:
        SensorPrediction with activity label, confidence, all class scores,
        and inference latency.
    """
    # Preprocess: gravity removal + normalization → (1, 128, 9)
    tensor = preprocess_sensor_window(total_acc, gyro, bundle)

    # Inference
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = bundle.model(tensor)          # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0]    # (num_classes,)
    inference_ms = (time.perf_counter() - t0) * 1000

    probs_list = probs.cpu().tolist()
    predicted_idx = int(probs.argmax().item())

    return SensorPrediction(
        activity=bundle.activity_names[predicted_idx],
        confidence=round(probs_list[predicted_idx], 4),
        all_scores={
            name: round(score, 4)
            for name, score in zip(bundle.activity_names, probs_list)
        },
        inference_ms=round(inference_ms, 2),
    )

