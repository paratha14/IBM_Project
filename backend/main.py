"""
main.py
-------
FastAPI application entry point for the Human Activity Tracker backend.

Startup sequence:
  1. Both ML models are loaded from their .pkl checkpoints (lifespan).
  2. CORS middleware is added with open origins so any frontend
     (web app, PWA, mobile browser, React Native, etc.) can connect.
  3. Routers for sensor and video endpoints are registered.
  4. A /health endpoint confirms the server is ready.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_DESCRIPTION, API_TITLE, API_VERSION
from backend.models.loader import get_sensor_bundle, get_video_bundle, load_all_models
from backend.routers import sensor, video
# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Track server start time for uptime reporting
# ---------------------------------------------------------------------------
_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Lifespan — runs at startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load both ML models before the server starts accepting requests.
    Any error here will prevent the app from starting — intentional,
    because running without models is useless.
    """
    logger.info("=== Human Activity Tracker API starting up ===")
    load_all_models()
    logger.info("=== Models ready. Server is accepting requests. ===")
    yield
    logger.info("=== Shutting down. ===")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — open to all origins so any web app / PWA / mobile browser can connect
#
# When you build the web app later, you can tighten this to only allow your
# specific frontend origin by replacing ["*"] with e.g. ["http://localhost:3000"].
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Sensor HTTP endpoints : POST /sensor/predict, POST /sensor/push
app.include_router(sensor.router)

# WebSocket: WS /ws/stream/{device_id}
app.include_router(sensor.ws_router)

# Video endpoint: POST /video/predict
app.include_router(video.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health() -> dict:
    """
    Returns server status and model metadata.
    Use this to verify the server is running and models are loaded.
    """
    sb = get_sensor_bundle()
    vb = get_video_bundle()

    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "models": {
            "sensor": {
                "name": sb.model_name,
                "classes": sb.activity_names,
                "num_classes": len(sb.activity_names),
                "validation_accuracy_pct": round(sb.val_accuracy * 100, 2),
                "test_accuracy_pct": round(sb.test_accuracy * 100, 2),
                "input_shape": "batch × 128 time_steps × 9 channels",
                "sample_rate_hz": 50,
                "window_duration_s": 2.56,
            },
            "video": {
                "name": "PoseMotionBiLSTM",
                "classes": vb.action_names,
                "num_classes": len(vb.action_names),
                "test_accuracy_pct": round(vb.test_accuracy * 100, 2),
                "input_shape": "batch × 64 frames × 91 pose_features",
            },
        },
        "endpoints": {
            "sensor_predict":   "POST /sensor/predict",
            "sensor_push":      "POST /sensor/push",
            "sensor_stream":    "WS   /ws/stream/{device_id}",
            "video_predict":    "POST /video/predict",
            "docs":             "GET  /docs",
        },
    }


# ---------------------------------------------------------------------------
# Root redirect to docs
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
