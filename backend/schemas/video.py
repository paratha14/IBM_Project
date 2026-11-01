"""
schemas/video.py
----------------
Pydantic response models for the video prediction endpoint.
Request is a multipart file upload (handled by FastAPI directly),
so no request schema is needed here.
"""

from pydantic import BaseModel


class VideoPredictResponse(BaseModel):
    activity: str
    confidence: float
    all_scores: dict[str, float]
    inference_ms: float
    preprocessing_ms: float
    total_ms: float
    model: str = "PoseMotionBiLSTM"
    note: str = (
        "Activities are from the Penn Action dataset: baseball_pitch, "
        "baseball_swing, bench_press, bowl, clean_and_jerk, golf_swing, "
        "jump_rope, jumping_jacks, pullup, pushup, situp, squat, "
        "strum_guitar, tennis_forehand, tennis_serve."
    )


class VideoErrorResponse(BaseModel):
    error: str
    detail: str

