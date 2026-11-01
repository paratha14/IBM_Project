"""
services/video_predictor.py
----------------------------
Runs inference with the video pose model and returns a structured result.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from backend.models.loader import VideoModelBundle
from backend.services.video_preprocessor import NoPoseDetectedError, preprocess_video


@dataclass
class VideoPrediction:
    activity: str                   # predicted action label
    confidence: float               # softmax probability of predicted class
    all_scores: dict[str, float]    # softmax probabilities for every class
    inference_ms: float             # model inference time (ms)
    preprocessing_ms: float         # pose extraction + feature engineering (ms)


def predict_from_video(
    video_bytes: bytes,
    bundle: VideoModelBundle,
) -> VideoPrediction:
    """
    Run the full video prediction pipeline (pose extraction → inference).

    Args:
        video_bytes: raw bytes of the uploaded video file.
        bundle:      loaded VideoModelBundle.

    Returns:
        VideoPrediction with activity label, confidence, all scores, and timings.

    Raises:
        NoPoseDetectedError: if no human pose is found in the video.
    """
    # --- Step 1: Pose extraction + feature engineering ---
    t_pre_start = time.perf_counter()
    tensor = preprocess_video(video_bytes, bundle)   # (1, 64, 91)
    preprocessing_ms = (time.perf_counter() - t_pre_start) * 1000

    # --- Step 2: Model inference ---
    t_inf_start = time.perf_counter()
    with torch.inference_mode():
        logits = bundle.model(tensor)               # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0]         # (num_classes,)
    inference_ms = (time.perf_counter() - t_inf_start) * 1000

    probs_list = probs.cpu().tolist()
    predicted_idx = int(probs.argmax().item())

    return VideoPrediction(
        activity=bundle.action_names[predicted_idx],
        confidence=round(probs_list[predicted_idx], 4),
        all_scores={
            name: round(score, 4)
            for name, score in zip(bundle.action_names, probs_list)
        },
        inference_ms=round(inference_ms, 2),
        preprocessing_ms=round(preprocessing_ms, 2),
    )

