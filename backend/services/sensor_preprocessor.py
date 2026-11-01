"""
services/sensor_preprocessor.py
--------------------------------
Signal processing pipeline that converts raw phone sensor readings
into the 9-channel normalized tensor expected by the sensor HAR model.

Raw phone data provides:
  - total_acc_x / y / z   (accelerometer, gravity included)
  - gyro_x / y / z        (gyroscope)

The UCI HAR model additionally needs:
  - body_acc_x / y / z    (total_acc minus gravity component)

Gravity is extracted via a 3rd-order Butterworth low-pass filter at 0.3 Hz,
matching the preprocessing described in the UCI HAR dataset documentation.

Channel ordering in the output (MUST match SENSOR_CHANNELS in config.py):
  0  body_acc_x
  1  body_acc_y
  2  body_acc_z
  3  body_gyro_x
  4  body_gyro_y
  5  body_gyro_z
  6  total_acc_x
  7  total_acc_y
  8  total_acc_z
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from backend.config import (
    GRAVITY_CUTOFF_HZ,
    GRAVITY_FILTER_ORDER,
    SENSOR_SAMPLE_RATE,
    SENSOR_WINDOW_SIZE,
)
from backend.models.loader import SensorModelBundle


# ---------------------------------------------------------------------------
# Butterworth low-pass filter
# ---------------------------------------------------------------------------

def _butter_lowpass_coeffs(
    cutoff_hz: float = GRAVITY_CUTOFF_HZ,
    fs: float = SENSOR_SAMPLE_RATE,
    order: int = GRAVITY_FILTER_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (b, a) Butterworth filter coefficients."""
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return b, a


# Pre-compute filter coefficients once at import time
_B, _A = _butter_lowpass_coeffs()


def _extract_gravity(signal: np.ndarray) -> np.ndarray:
    """
    Apply a zero-phase (forward-backward) Butterworth low-pass filter to
    a 1-D signal and return the estimated gravity component.

    Args:
        signal: 1-D array of shape (n_samples,) — one axis of total_acc.

    Returns:
        gravity: 1-D array of same shape.
    """
    return filtfilt(_B, _A, signal).astype(np.float32)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_sensor_window(
    total_acc: np.ndarray,
    gyro: np.ndarray,
    bundle: SensorModelBundle,
) -> torch.Tensor:
    """
    Convert a raw 128-sample window into a normalized model-ready tensor.

    Args:
        total_acc: shape (128, 3) — columns are [x, y, z] total acceleration (m/s²)
        gyro:      shape (128, 3) — columns are [x, y, z] angular velocity (rad/s)
        bundle:    SensorModelBundle containing mean and std from checkpoint.

    Returns:
        Tensor of shape (1, 128, 9) on the device specified by bundle.device,
        ready to be passed directly to the model.

    Raises:
        ValueError: if input arrays have unexpected shapes.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if total_acc.shape != (SENSOR_WINDOW_SIZE, 3):
        raise ValueError(
            f"total_acc must have shape ({SENSOR_WINDOW_SIZE}, 3), "
            f"got {total_acc.shape}"
        )
    if gyro.shape != (SENSOR_WINDOW_SIZE, 3):
        raise ValueError(
            f"gyro must have shape ({SENSOR_WINDOW_SIZE}, 3), "
            f"got {gyro.shape}"
        )

    total_acc = total_acc.astype(np.float32)
    gyro = gyro.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 1 — Extract gravity and compute body acceleration
    # ------------------------------------------------------------------
    gravity = np.stack(
        [_extract_gravity(total_acc[:, axis]) for axis in range(3)],
        axis=1,
    )  # (128, 3)

    body_acc = total_acc - gravity  # (128, 3)

    # ------------------------------------------------------------------
    # Step 2 — Stack channels in the required order
    #   [body_acc_x, body_acc_y, body_acc_z,
    #    body_gyro_x, body_gyro_y, body_gyro_z,
    #    total_acc_x, total_acc_y, total_acc_z]
    # ------------------------------------------------------------------
    window = np.concatenate([body_acc, gyro, total_acc], axis=1)  # (128, 9)

    # ------------------------------------------------------------------
    # Step 3 — Normalize using training-time statistics stored in checkpoint
    # ------------------------------------------------------------------
    # bundle.mean / std have shape (1, 1, 9) — broadcast over (1, 128, 9)
    tensor = torch.from_numpy(window).unsqueeze(0)  # (1, 128, 9)
    tensor = tensor.to(bundle.device)
    tensor = (tensor - bundle.mean) / bundle.std     # z-score normalization

    return tensor.float()

