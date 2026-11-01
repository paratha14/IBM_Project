"""
schemas/sensor.py
-----------------
Pydantic request and response models for the sensor (HAR) endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SensorSample(BaseModel):
    """A single raw sensor reading from the phone at one point in time."""
    timestamp: float = Field(..., description="Unix timestamp (seconds) of this sample")
    total_acc_x: float = Field(..., description="Total acceleration X axis (m/s²)")
    total_acc_y: float = Field(..., description="Total acceleration Y axis (m/s²)")
    total_acc_z: float = Field(..., description="Total acceleration Z axis (m/s²)")
    gyro_x: float = Field(..., description="Gyroscope X axis (rad/s)")
    gyro_y: float = Field(..., description="Gyroscope Y axis (rad/s)")
    gyro_z: float = Field(..., description="Gyroscope Z axis (rad/s)")


class ActivityScores(BaseModel):
    """Softmax probability for each activity class."""
    WALKING: float
    WALKING_UPSTAIRS: float
    WALKING_DOWNSTAIRS: float
    SITTING: float
    STANDING: float
    LAYING: float


# ---------------------------------------------------------------------------
# POST /sensor/predict  —  full-window inference
# ---------------------------------------------------------------------------

class SensorWindow(BaseModel):
    """
    A complete 128-sample window sent in one request.
    Each list must contain exactly 128 values (50 Hz × 2.56 s).
    """
    total_acc_x: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_y: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_z: list[float] = Field(..., min_length=128, max_length=128)
    gyro_x: list[float] = Field(..., min_length=128, max_length=128)
    gyro_y: list[float] = Field(..., min_length=128, max_length=128)
    gyro_z: list[float] = Field(..., min_length=128, max_length=128)


class SensorPredictRequest(BaseModel):
    device_id: str = Field(default="default", description="Unique identifier for the device/user")
    timestamp: Optional[float] = Field(default=None, description="Window start time (Unix seconds)")
    window: SensorWindow


class SensorPredictResponse(BaseModel):
    device_id: str
    activity: str
    confidence: float
    all_scores: dict[str, float]
    inference_ms: float
    model: str = "Temporal CNN-BiLSTM"


# ---------------------------------------------------------------------------
# POST /sensor/push  —  incremental single-sample push
# ---------------------------------------------------------------------------

class SensorPushRequest(BaseModel):
    device_id: str = Field(default="default")
    timestamp: float
    total_acc_x: float
    total_acc_y: float
    total_acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class SensorPushResponse(BaseModel):
    """
    Returned after each pushed sample.
    When the buffer has collected enough samples a prediction is included;
    otherwise only the buffer status is returned.
    """
    device_id: str
    status: str                           # "buffering" or "predicted"
    buffer_size: int
    window_size: int = 128
    # Only present when status == "predicted"
    activity: Optional[str] = None
    confidence: Optional[float] = None
    all_scores: Optional[dict[str, float]] = None
    inference_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# WebSocket /ws/stream/{device_id}  —  real-time streaming message schemas
# (used for JSON serialization / deserialization over the WebSocket)
# ---------------------------------------------------------------------------

class WSIncomingSample(BaseModel):
    """JSON message the phone sends over the WebSocket (one sample at a time)."""
    timestamp: float
    total_acc_x: float
    total_acc_y: float
    total_acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class WSPredictionMessage(BaseModel):
    """JSON message the server pushes back to the phone after each prediction."""
    type: str = "prediction"
    activity: str
    confidence: float
    all_scores: dict[str, float]
    inference_ms: float
    buffer_size: int


class WSStatusMessage(BaseModel):
    """Acknowledgement sent while the buffer is still filling up."""
    type: str = "buffering"
    buffer_size: int
    window_size: int = 128
    needed: int                           # samples still needed before next prediction

