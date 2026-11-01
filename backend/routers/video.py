"""
routers/video.py
----------------
Video activity recognition endpoint.

  POST /video/predict  — Upload a video file, get predicted action back.

The video is processed entirely server-side:
  1. MediaPipe extracts pose landmarks frame-by-frame.
  2. Landmarks are resampled to 64 frames and converted to 91 features.
  3. The PoseMotionBiLSTM model classifies the action.
  4. Result is returned as JSON.

Supported formats: MP4, AVI, MOV (anything OpenCV can decode).
Max upload size: 100 MB (configurable in config.py).
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.config import ALLOWED_VIDEO_TYPES, MAX_VIDEO_SIZE_BYTES
from backend.models.loader import get_video_bundle
from backend.schemas.video import VideoPredictResponse
from backend.services.video_predictor import predict_from_video
from backend.services.video_preprocessor import NoPoseDetectedError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video", tags=["Video HAR"])

# Check if python-multipart is installed
try:
    import multipart  # python-multipart package provides the 'multipart' module
    _MULTIPART_AVAILABLE = True
except ImportError:
    _MULTIPART_AVAILABLE = False


def _process_video_bytes(video_bytes: bytes, filename: str = "video.mp4") -> VideoPredictResponse:
    bundle = get_video_bundle()

    if len(video_bytes) > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(video_bytes) / 1_048_576:.1f} MB "
                f"exceeds the {MAX_VIDEO_SIZE_BYTES // 1_048_576} MB limit."
            ),
        )

    if len(video_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info("video/predict | filename=%s | size=%.2f MB", filename, len(video_bytes) / 1_048_576)

    try:
        prediction = predict_from_video(video_bytes, bundle)
    except NoPoseDetectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Video prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    logger.info(
        "video/predict | activity=%s | conf=%.2f%% | preprocess=%.0fms | inference=%.0fms",
        prediction.activity,
        prediction.confidence * 100,
        prediction.preprocessing_ms,
        prediction.inference_ms,
    )

    return VideoPredictResponse(
        activity=prediction.activity,
        confidence=prediction.confidence,
        all_scores=prediction.all_scores,
        inference_ms=prediction.inference_ms,
        preprocessing_ms=prediction.preprocessing_ms,
        total_ms=round(prediction.inference_ms + prediction.preprocessing_ms, 2),
    )


if _MULTIPART_AVAILABLE:
    from fastapi import File, UploadFile

    @router.post(
        "/predict",
        response_model=VideoPredictResponse,
        summary="Predict action from an uploaded video file (multipart form-data)",
    )
    async def video_predict_form(
        file: UploadFile = File(..., description="Video file (MP4, AVI, MOV). Max 100 MB.")
    ) -> VideoPredictResponse:
        content_type = file.content_type or ""
        if content_type and content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: '{content_type}'. Accepted types: {sorted(ALLOWED_VIDEO_TYPES)}",
            )
        video_bytes = await file.read()
        return _process_video_bytes(video_bytes, file.filename or "video.mp4")

else:
    @router.post(
        "/predict",
        response_model=VideoPredictResponse,
        summary="Predict action from raw video bytes (install python-multipart for form-data)",
    )
    async def video_predict_raw(request: Request) -> VideoPredictResponse:
        """
        Accepts raw video bytes in the request body (Content-Type: video/mp4).
        Install `python-multipart` to enable standard browser multipart/form-data upload.
        """
        video_bytes = await request.body()
        return _process_video_bytes(video_bytes, "raw_stream.mp4")
