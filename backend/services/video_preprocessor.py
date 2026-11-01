"""
services/video_preprocessor.py
--------------------------------
Converts an uploaded video file into the 91-feature pose sequence
expected by the PoseMotionBiLSTM model.

Pipeline:
  1. Read all frames from the video using OpenCV.
  2. Run MediaPipe Pose on each frame to get 33 body landmarks.
  3. Map the 33 MediaPipe landmarks → 13 Penn Action joints.
  4. Resample the sequence to exactly VIDEO_SEQ_LEN (64) frames via
     linear interpolation (same method used during training).
  5. Normalize joint coordinates by frame width / height.
  6. Compute per-frame features:
       - 26 normalized joint (x, y) coordinates
       - 13 joint visibility scores
       - 26 frame-to-frame velocity vectors (Δx, Δy)
       - 26 parent-relative bone vectors
  7. Apply checkpoint mean / std normalization.
  8. Return (1, 64, 91) float tensor ready for model inference.
"""

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.config import (
    MEDIAPIPE_TO_PENN_JOINTS,
    PENN_JOINT_PARENTS,
    PENN_TORSO_JOINTS,
    VIDEO_INPUT_SIZE,
    VIDEO_SEQ_LEN,
)
from backend.models.loader import VideoModelBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe lazy import (so the rest of the app works if mediapipe is absent or
# partially installed / incompatible in the deployment environment)
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise AttributeError("mediapipe.solutions is unavailable in this installation")

    _mp_pose = mp.solutions.pose
    _MEDIAPIPE_AVAILABLE = True
except Exception as exc:  # broad exception so startup does not fail on a broken install
    _MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "mediapipe is unavailable or incompatible in this environment; "
        "video prediction will not be available. Error: %s",
        exc,
    )


# ---------------------------------------------------------------------------
# Helper: extract MediaPipe landmarks from a single BGR frame
# ---------------------------------------------------------------------------

def _extract_landmarks_from_frame(
    pose_instance,
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Run MediaPipe Pose on one BGR frame.

    Returns:
        x_coords  : (13,) float32 — pixel x for each Penn joint
        y_coords  : (13,) float32 — pixel y for each Penn joint
        visibility: (13,) float32 — confidence score per joint
        width, height: frame dimensions (for normalisation)
    """
    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose_instance.process(rgb)

    x_coords = np.zeros(13, dtype=np.float32)
    y_coords = np.zeros(13, dtype=np.float32)
    visibility = np.zeros(13, dtype=np.float32)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for penn_idx, mp_idx in MEDIAPIPE_TO_PENN_JOINTS.items():
            lm = landmarks[mp_idx]
            x_coords[penn_idx] = lm.x * width    # convert normalised → pixel
            y_coords[penn_idx] = lm.y * height
            visibility[penn_idx] = lm.visibility

    return x_coords, y_coords, visibility, width, height


# ---------------------------------------------------------------------------
# Helper: build 91 features from raw joint arrays (mirrors the training code)
# ---------------------------------------------------------------------------

def _build_features(
    x_seq: np.ndarray,   # (T, 13) pixel x
    y_seq: np.ndarray,   # (T, 13) pixel y
    vis_seq: np.ndarray, # (T, 13) visibility
    width: float,
    height: float,
) -> np.ndarray:
    """
    Replicate the feature engineering from the training notebook.

    Returns:
        features: (T, 91) float32
    """
    T = x_seq.shape[0]

    # Normalize coordinates to [0, 1] range (zero out invisible joints)
    x_norm = (x_seq / max(width, 1.0)) * (vis_seq > 0).astype(np.float32)
    y_norm = (y_seq / max(height, 1.0)) * (vis_seq > 0).astype(np.float32)

    # (T, 13, 2) coordinate array
    coordinates = np.stack((x_norm, y_norm), axis=-1)  # (T, 13, 2)

    # Frame-to-frame velocity
    velocity = np.zeros_like(coordinates)
    velocity[1:] = coordinates[1:] - coordinates[:-1]

    # Bone vectors (parent-relative)
    # Torso joints used to compute the torso centroid
    torso_weights = vis_seq[:, PENN_TORSO_JOINTS, None]        # (T, 4, 1)
    torso_coords = coordinates[:, PENN_TORSO_JOINTS, :]        # (T, 4, 2)
    torso_center = (
        (torso_coords * torso_weights).sum(axis=1)
        / (torso_weights.sum(axis=1) + 1e-8)
    )  # (T, 2)

    bones = np.zeros_like(coordinates)  # (T, 13, 2)
    for joint_idx, parent_idx in enumerate(PENN_JOINT_PARENTS):
        if parent_idx == -1:
            # Root joints: relative to torso center
            bones[:, joint_idx, :] = (
                coordinates[:, joint_idx, :] - torso_center
            ) * vis_seq[:, joint_idx, None]
        else:
            # Other joints: relative to their parent
            bones[:, joint_idx, :] = (
                coordinates[:, joint_idx, :] - coordinates[:, parent_idx, :]
            ) * (
                vis_seq[:, joint_idx] * vis_seq[:, parent_idx]
            )[:, None]

    # Concatenate all features: (T, 26 + 13 + 26 + 26) = (T, 91)
    features = np.concatenate(
        [
            coordinates.reshape(T, 26),  # joint x, y
            vis_seq.reshape(T, 13),       # visibility
            velocity.reshape(T, 26),      # frame velocities
            bones.reshape(T, 26),         # bone vectors
        ],
        axis=1,
    ).astype(np.float32)

    assert features.shape == (T, VIDEO_INPUT_SIZE), (
        f"Feature shape mismatch: expected ({T}, {VIDEO_INPUT_SIZE}), "
        f"got {features.shape}"
    )
    return features


# ---------------------------------------------------------------------------
# Helper: linear resample to exactly VIDEO_SEQ_LEN frames
# ---------------------------------------------------------------------------

def _resample_to_seq_len(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    vis_seq: np.ndarray,
    target_len: int = VIDEO_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearly interpolate pose sequences to exactly `target_len` frames.
    Mirrors the vectorised resampling used in the training notebook.
    """
    n_frames = len(x_seq)
    if n_frames == 0:
        raise ValueError("No frames with pose data were found in the video.")

    positions = np.linspace(0, n_frames - 1, target_len)
    left = positions.astype(np.int32)
    right = np.minimum(left + 1, n_frames - 1)
    weight = (positions - left)[:, None]  # (target_len, 1)

    x_resampled = x_seq[left] * (1 - weight) + x_seq[right] * weight
    y_resampled = y_seq[left] * (1 - weight) + y_seq[right] * weight
    vis_resampled = vis_seq[left] * (1 - weight) + vis_seq[right] * weight

    return x_resampled, y_resampled, vis_resampled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NoPoseDetectedError(Exception):
    """Raised when MediaPipe cannot detect a person in the video."""


def preprocess_video(
    video_bytes: bytes,
    bundle: VideoModelBundle,
) -> torch.Tensor:
    """
    Convert raw video bytes into the normalized pose tensor expected by
    PoseMotionBiLSTM.

    Args:
        video_bytes: raw bytes of an uploaded video file (MP4, AVI, MOV…)
        bundle:      VideoModelBundle with mean/std from checkpoint

    Returns:
        Tensor of shape (1, 64, 91) on bundle.device, ready for inference.

    Raises:
        NoPoseDetectedError: if MediaPipe finds no person in any frame.
        RuntimeError:        if mediapipe is not installed.
    """
    if not _MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            "mediapipe is not installed. "
            "Run: pip install mediapipe"
        )

    # Write bytes to a temp file — OpenCV needs a file path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open the video file: {tmp_path}")

        all_x, all_y, all_vis = [], [], []
        frame_width = frame_height = 1

        with _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                x_c, y_c, vis_c, fw, fh = _extract_landmarks_from_frame(pose, frame)
                all_x.append(x_c)
                all_y.append(y_c)
                all_vis.append(vis_c)
                frame_width = fw
                frame_height = fh

        cap.release()
    finally:
        tmp_path.unlink(missing_ok=True)

    if len(all_x) == 0:
        raise NoPoseDetectedError("No frames could be read from the video.")

    x_seq = np.stack(all_x)     # (N, 13)
    y_seq = np.stack(all_y)     # (N, 13)
    vis_seq = np.stack(all_vis) # (N, 13)

    # Check that at least some joints were detected
    if vis_seq.max() == 0:
        raise NoPoseDetectedError(
            "MediaPipe could not detect a person in any frame of the video."
        )

    # Resample → (64, 13) for each array
    x_seq, y_seq, vis_seq = _resample_to_seq_len(x_seq, y_seq, vis_seq)

    logger.debug(
        "Video processed: %d frames resampled to %d | frame size: %dx%d",
        len(all_x),
        VIDEO_SEQ_LEN,
        frame_width,
        frame_height,
    )

    # Build 91 features → (64, 91)
    features = _build_features(x_seq, y_seq, vis_seq, frame_width, frame_height)

    # Normalize using checkpoint statistics → (1, 64, 91)
    tensor = torch.from_numpy(features).unsqueeze(0).to(bundle.device)
    tensor = (tensor - bundle.mean) / bundle.std

    return tensor.float()

