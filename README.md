# 🏃 Human Activity Tracker — Backend & API Documentation

FastAPI backend for real-time human activity recognition using deep learning models:
1. **Sensor Model (HAR)**: 6 activities (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) from smartphone accelerometer + gyroscope readings.
2. **Video Model (Penn Action)**: 15 actions (`pushup`, `pullup`, `squat`, `jump_rope`, `baseball_swing`, etc.) from uploaded video clips.

---

## 🛠️ Quick Setup for Frontend Teammate

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Start the Backend Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## 📱 Frontend Integration Guide

CORS is enabled for all origins (`*`), so you can connect from React, Vue, React Native, Flutter, or vanilla HTML/JS directly.

### Option A: Real-Time Mobile Streaming (WebSocket) ⚡ [Recommended]

Connect your mobile web app (or mobile app) to:
```
ws://<SERVER_IP>:8000/ws/stream/<device_id>
```

#### What your frontend should send (JSON at ~50 Hz):
```json
{
  "timestamp": 1746100000.02,
  "total_acc_x": 0.12,
  "total_acc_y": -0.03,
  "total_acc_z": 9.81,
  "gyro_x": 0.01,
  "gyro_y": -0.02,
  "gyro_z": 0.00
}
```

#### What your frontend receives:
While filling the initial buffer (first 128 samples):
```json
{
  "type": "buffering",
  "buffer_size": 45,
  "window_size": 128,
  "needed": 83
}
```
When a prediction is made (every ~1.28 seconds over the sliding window):
```json
{
  "type": "prediction",
  "activity": "WALKING",
  "confidence": 0.9412,
  "all_scores": {
    "WALKING": 0.9412,
    "WALKING_UPSTAIRS": 0.0315,
    "WALKING_DOWNSTAIRS": 0.0120,
    "SITTING": 0.0081,
    "STANDING": 0.0052,
    "LAYING": 0.0020
  },
  "inference_ms": 14.2,
  "buffer_size": 128
}
```

---

### Option B: HTTP Incremental Push (`POST /sensor/push`)

Send individual sensor samples over standard HTTP:
- **URL**: `POST http://<SERVER_IP>:8000/sensor/push`
- **Body**: Same JSON payload as WebSocket.
- **Response**: Returns `{"status": "buffering", ...}` or `{"status": "predicted", "activity": "WALKING", ...}`.

---

### Option C: Video Upload Prediction (`POST /video/predict`)

- **URL**: `POST http://<SERVER_IP>:8000/video/predict`
- **Method**: Multipart Form-Data (or raw video bytes) with field `file` containing an MP4/AVI/MOV video.
- **Response**:
```json
{
  "activity": "pushup",
  "confidence": 0.9234,
  "all_scores": { "pushup": 0.9234, "pullup": 0.0411, ... },
  "inference_ms": 28.5,
  "preprocessing_ms": 320.1,
  "total_ms": 348.6,
  "model": "PoseMotionBiLSTM"
}
```

---

## 🧪 Running Backend Automated Tests

In a separate terminal while the server is running:
```bash
python test_backend.py
```
This tests health check, batch prediction, push buffer, and live WebSocket streaming.

