"""
routers/sensor.py
-----------------
All sensor-related API endpoints:

  POST /sensor/predict   — Send a complete 128-sample window, get activity back.
  POST /sensor/push      — Push one sample at a time; server buffers and predicts.
  WS   /ws/stream/{id}  — Bidirectional real-time streaming (recommended for web app).

WebSocket protocol (JSON over text frames):
  Phone → Server : WSIncomingSample  (one per sample, ~50 Hz)
  Server → Phone : WSPredictionMessage  (every ~1.28 s when buffer is ready)
                   WSStatusMessage      (every sample while buffer is filling)
"""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from backend.config import SENSOR_WINDOW_SIZE
from backend.models.loader import get_sensor_bundle
from backend.schemas.sensor import (
    SensorPredictRequest,
    SensorPredictResponse,
    SensorPushRequest,
    SensorPushResponse,
    WSIncomingSample,
    WSPredictionMessage,
    WSStatusMessage,
)
from backend.services.buffer import sensor_buffer
from backend.services.sensor_predictor import predict_from_window

logger = logging.getLogger(__name__)

# HTTP routes live under /sensor
router = APIRouter(prefix="/sensor", tags=["Sensor HAR"])

# WebSocket router — separate prefix so WS lives at /ws/stream/{device_id}
ws_router = APIRouter(prefix="/ws", tags=["WebSocket Stream"])


# ---------------------------------------------------------------------------
# Helper — build numpy arrays from a window_data dict
# ---------------------------------------------------------------------------

def _window_dict_to_numpy(window_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the channel-keyed dict produced by the buffer / request schemas
    into (total_acc, gyro) numpy arrays of shape (128, 3).
    """
    total_acc = np.column_stack([
        window_data["total_acc_x"],
        window_data["total_acc_y"],
        window_data["total_acc_z"],
    ]).astype(np.float32)

    gyro = np.column_stack([
        window_data["gyro_x"],
        window_data["gyro_y"],
        window_data["gyro_z"],
    ]).astype(np.float32)

    return total_acc, gyro


# ---------------------------------------------------------------------------
# POST /sensor/predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=SensorPredictResponse,
    summary="Predict activity from a full 128-sample sensor window",
)
async def sensor_predict(request: SensorPredictRequest) -> SensorPredictResponse:
    """
    Accepts a complete 128-sample window (2.56 s @ 50 Hz) and returns the
    predicted activity with confidence scores.

    **Use this endpoint when:**
    - The phone collects a full window locally before sending.
    - You want a one-shot REST call (no persistent connection required).
    """
    bundle = get_sensor_bundle()

    window = request.window
    window_data = {
        "total_acc_x": window.total_acc_x,
        "total_acc_y": window.total_acc_y,
        "total_acc_z": window.total_acc_z,
        "gyro_x":      window.gyro_x,
        "gyro_y":      window.gyro_y,
        "gyro_z":      window.gyro_z,
    }

    total_acc, gyro = _window_dict_to_numpy(window_data)

    try:
        prediction = predict_from_window(total_acc, gyro, bundle)
    except Exception as exc:
        logger.exception("Sensor prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    logger.info(
        "sensor/predict | device=%s | activity=%s | conf=%.2f%%",
        request.device_id,
        prediction.activity,
        prediction.confidence * 100,
    )

    return SensorPredictResponse(
        device_id=request.device_id,
        activity=prediction.activity,
        confidence=prediction.confidence,
        all_scores=prediction.all_scores,
        inference_ms=prediction.inference_ms,
    )


# ---------------------------------------------------------------------------
# POST /sensor/push
# ---------------------------------------------------------------------------

@router.post(
    "/push",
    response_model=SensorPushResponse,
    summary="Push one sensor sample; server buffers and predicts automatically",
)
async def sensor_push(request: SensorPushRequest) -> SensorPushResponse:
    """
    Push individual sensor samples at ~50 Hz.  
    The server accumulates samples in a per-device rolling window buffer and 
    returns a prediction once every 64 new samples (sliding window, 50 % overlap).

    **Use this endpoint when:**
    - You prefer simple HTTP POST calls over WebSocket.
    - You're building a quick prototype or testing.
    """
    bundle = get_sensor_bundle()
    device_id = request.device_id

    sample = {
        "total_acc_x": request.total_acc_x,
        "total_acc_y": request.total_acc_y,
        "total_acc_z": request.total_acc_z,
        "gyro_x":      request.gyro_x,
        "gyro_y":      request.gyro_y,
        "gyro_z":      request.gyro_z,
    }

    ready, window_data = sensor_buffer.push(device_id, sample)
    buf_size = sensor_buffer.buffer_size(device_id)

    if not ready:
        return SensorPushResponse(
            device_id=device_id,
            status="buffering",
            buffer_size=buf_size,
        )

    # Window is ready — run prediction
    total_acc, gyro = _window_dict_to_numpy(window_data)

    try:
        prediction = predict_from_window(total_acc, gyro, bundle)
    except Exception as exc:
        logger.exception("Sensor push prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    logger.info(
        "sensor/push | device=%s | activity=%s | conf=%.2f%%",
        device_id,
        prediction.activity,
        prediction.confidence * 100,
    )

    return SensorPushResponse(
        device_id=device_id,
        status="predicted",
        buffer_size=buf_size,
        activity=prediction.activity,
        confidence=prediction.confidence,
        all_scores=prediction.all_scores,
        inference_ms=prediction.inference_ms,
    )


# ---------------------------------------------------------------------------
# WebSocket /ws/stream/{device_id}
# ---------------------------------------------------------------------------

@ws_router.websocket("/stream/{device_id}")
async def sensor_stream(websocket: WebSocket, device_id: str) -> None:
    """
    Real-time bidirectional WebSocket stream.

    **Phone → Server**: JSON object matching WSIncomingSample (one per sensor tick).
    **Server → Phone**: JSON object — WSStatusMessage while buffering,
                        WSPredictionMessage when a prediction is ready.

    The server uses a 128-sample sliding window (50 % overlap) and pushes
    a prediction approximately every 1.28 seconds of continuous streaming.

    **This is the recommended endpoint for the web app / PWA.**
    """
    await websocket.accept()
    logger.info("WebSocket connected: device_id=%s", device_id)
    bundle = get_sensor_bundle()

    # Clear any stale buffer from a previous session
    sensor_buffer.clear(device_id)

    try:
        while True:
            # Receive one sample from the phone
            raw = await websocket.receive_text()

            try:
                sample_msg = WSIncomingSample.model_validate_json(raw)
            except Exception as parse_err:
                await websocket.send_text(
                    f'{{"type":"error","detail":"Invalid JSON: {parse_err}"}}'
                )
                continue

            sample = {
                "total_acc_x": sample_msg.total_acc_x,
                "total_acc_y": sample_msg.total_acc_y,
                "total_acc_z": sample_msg.total_acc_z,
                "gyro_x":      sample_msg.gyro_x,
                "gyro_y":      sample_msg.gyro_y,
                "gyro_z":      sample_msg.gyro_z,
            }

            ready, window_data = sensor_buffer.push(device_id, sample)
            buf_size = sensor_buffer.buffer_size(device_id)

            if not ready:
                # Still buffering — send a status ack so the phone knows it's alive
                status_msg = WSStatusMessage(
                    buffer_size=buf_size,
                    needed=max(SENSOR_WINDOW_SIZE - buf_size, 0),
                )
                await websocket.send_text(status_msg.model_dump_json())
                continue

            # Prediction window ready
            total_acc, gyro = _window_dict_to_numpy(window_data)

            try:
                prediction = predict_from_window(total_acc, gyro, bundle)
            except Exception as exc:
                logger.exception("WS inference failed: %s", exc)
                await websocket.send_text(
                    f'{{"type":"error","detail":"Inference failed: {exc}"}}'
                )
                continue

            pred_msg = WSPredictionMessage(
                activity=prediction.activity,
                confidence=prediction.confidence,
                all_scores=prediction.all_scores,
                inference_ms=prediction.inference_ms,
                buffer_size=buf_size,
            )
            await websocket.send_text(pred_msg.model_dump_json())

            logger.info(
                "WS stream | device=%s | activity=%s | conf=%.2f%%",
                device_id,
                prediction.activity,
                prediction.confidence * 100,
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: device_id=%s", device_id)
    finally:
        sensor_buffer.clear(device_id)
