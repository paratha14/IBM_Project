"""
models/loader.py
----------------
Loads and caches both ML models at application startup.

Usage (called once from main.py lifespan):
    from backend.models.loader import load_all_models, get_sensor_bundle, get_video_bundle

Both models are stored as "bundles" — a named tuple that contains the
PyTorch model plus all metadata (class names, normalization stats, etc.)
so every downstream service receives a self-contained object.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from backend.config import (
    SENSOR_MODEL_PATH,
    VIDEO_MODEL_PATH,
    SENSOR_HIDDEN_SIZE,
    SENSOR_INPUT_SIZE,
    SENSOR_NUM_CLASSES,
    VIDEO_HIDDEN_SIZE,
    VIDEO_INPUT_SIZE,
    VIDEO_NUM_CLASSES,
)
from backend.models.sensor_har import SENSOR_MODEL_REGISTRY
from backend.models.video_har import PoseMotionBiLSTM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundle dataclasses — carry model + metadata together
# ---------------------------------------------------------------------------

@dataclass
class SensorModelBundle:
    """Everything needed to run sensor inference."""
    model: torch.nn.Module
    mean: torch.Tensor          # shape (1, 1, 9)
    std: torch.Tensor           # shape (1, 1, 9)
    activity_names: list[str]
    channels: list[str]
    model_name: str
    val_accuracy: float
    test_accuracy: float
    device: torch.device


@dataclass
class VideoModelBundle:
    """Everything needed to run video pose inference."""
    model: torch.nn.Module
    mean: torch.Tensor          # shape (1, 1, 91)
    std: torch.Tensor           # shape (1, 1, 91)
    action_names: list[str]
    test_accuracy: float
    device: torch.device


# ---------------------------------------------------------------------------
# Module-level cache — populated by load_all_models()
# ---------------------------------------------------------------------------

_sensor_bundle: Optional[SensorModelBundle] = None
_video_bundle: Optional[VideoModelBundle] = None


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    """Use CUDA if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: %s", device)
    return device


def _load_sensor_model(device: torch.device) -> SensorModelBundle:
    """
    Load the sensor HAR checkpoint and reconstruct the correct model class.
    The checkpoint stores 'model_name' to tell us which architecture was selected.
    """
    logger.info("Loading sensor model from: %s", SENSOR_MODEL_PATH)

    checkpoint = torch.load(SENSOR_MODEL_PATH, map_location="cpu", weights_only=False)

    model_name: str = checkpoint["model_name"]
    model_class = SENSOR_MODEL_REGISTRY.get(model_name)

    if model_class is None:
        raise ValueError(
            f"Unknown sensor model name in checkpoint: '{model_name}'. "
            f"Expected one of: {list(SENSOR_MODEL_REGISTRY.keys())}"
        )

    model = model_class(
        input_size=checkpoint.get("input_size", SENSOR_INPUT_SIZE),
        hidden_size=checkpoint.get("hidden_size", SENSOR_HIDDEN_SIZE),
        num_classes=checkpoint.get("num_classes", SENSOR_NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    bundle = SensorModelBundle(
        model=model,
        mean=checkpoint["mean"].to(device),
        std=checkpoint["std"].to(device),
        activity_names=checkpoint["activity_names"],
        channels=checkpoint["channels"],
        model_name=model_name,
        val_accuracy=float(checkpoint.get("best_validation_accuracy", 0.0)),
        test_accuracy=float(checkpoint.get("test_accuracy", 0.0)),
        device=device,
    )

    logger.info(
        "Sensor model loaded: %s | val=%.2f%% | test=%.2f%%",
        model_name,
        bundle.val_accuracy * 100,
        bundle.test_accuracy * 100,
    )
    return bundle


def _load_video_model(device: torch.device) -> VideoModelBundle:
    """
    Load the Penn Action pose BiLSTM checkpoint.
    """
    logger.info("Loading video model from: %s", VIDEO_MODEL_PATH)

    checkpoint = torch.load(VIDEO_MODEL_PATH, map_location="cpu", weights_only=False)

    model = PoseMotionBiLSTM(
        input_size=checkpoint.get("input_size", VIDEO_INPUT_SIZE),
        hidden_size=checkpoint.get("hidden_size", VIDEO_HIDDEN_SIZE),
        num_classes=checkpoint.get("num_classes", VIDEO_NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    bundle = VideoModelBundle(
        model=model,
        mean=checkpoint["mean"].to(device),
        std=checkpoint["std"].to(device),
        action_names=checkpoint["action_names"],
        test_accuracy=float(checkpoint.get("test_accuracy", 0.0)),
        device=device,
    )

    logger.info(
        "Video model loaded | actions=%d | test=%.2f%%",
        len(bundle.action_names),
        bundle.test_accuracy * 100,
    )
    return bundle


def load_all_models() -> None:
    """
    Load and cache both models.
    Call this once during FastAPI's lifespan startup event.
    Raises on any loading failure so the app refuses to start with a broken model.
    """
    global _sensor_bundle, _video_bundle
    device = _select_device()
    _sensor_bundle = _load_sensor_model(device)
    _video_bundle = _load_video_model(device)
    logger.info("All models loaded and ready.")


# ---------------------------------------------------------------------------
# Accessors (used by routers / services)
# ---------------------------------------------------------------------------

def get_sensor_bundle() -> SensorModelBundle:
    if _sensor_bundle is None:
        raise RuntimeError("Sensor model is not loaded. Was load_all_models() called?")
    return _sensor_bundle


def get_video_bundle() -> VideoModelBundle:
    if _video_bundle is None:
        raise RuntimeError("Video model is not loaded. Was load_all_models() called?")
    return _video_bundle

