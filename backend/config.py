"""
config.py
---------
Central configuration for the Human Activity Tracker backend.
All file paths, model constants, and signal-processing parameters live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root & model paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root

SENSOR_MODEL_PATH = (
    BASE_DIR / "DL MODELS" / "Sensor model" / "har_bilstm_sensor_enhanced.pkl"
)
VIDEO_MODEL_PATH = (
    BASE_DIR
    / "DL MODELS"
    / "Video model"
    / "penn_action_pose_bilstm_91features (1).pkl"
)

# ---------------------------------------------------------------------------
# Sensor model constants  (must match training-time values in the notebook)
# ---------------------------------------------------------------------------
SENSOR_SAMPLE_RATE: int = 50          # Hz — UCI HAR was recorded at 50 Hz
SENSOR_WINDOW_SIZE: int = 128         # time steps per prediction window
SENSOR_SLIDE_SIZE: int = 64           # overlap stride (50 % overlap)
SENSOR_INPUT_SIZE: int = 9            # number of input channels
SENSOR_HIDDEN_SIZE: int = 64
SENSOR_NUM_CLASSES: int = 6

# Channel order — MUST match the order used during training
SENSOR_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

SENSOR_ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# Butterworth low-pass filter parameters for gravity extraction
# UCI HAR documentation specifies a 0.3 Hz cutoff, order 3
GRAVITY_CUTOFF_HZ: float = 0.3
GRAVITY_FILTER_ORDER: int = 3

# ---------------------------------------------------------------------------
# Video model constants  (must match training-time values in the notebook)
# ---------------------------------------------------------------------------
VIDEO_SEQ_LEN: int = 64              # frames sampled per clip
VIDEO_INPUT_SIZE: int = 91           # pose features per frame
VIDEO_HIDDEN_SIZE: int = 64
VIDEO_NUM_CLASSES: int = 15

# 15 Penn Action classes (alphabetically sorted — checkpoint ordering)
VIDEO_ACTION_NAMES = [
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowl",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pullup",
    "pushup",
    "situp",
    "squat",
    "strum_guitar",
    "tennis_forehand",
    "tennis_serve",
]

# MediaPipe → Penn Action joint index mapping
# Penn Action uses 13 joints; we pick the closest MediaPipe landmark indices
# MediaPipe landmark indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MEDIAPIPE_TO_PENN_JOINTS = {
    0: 0,   # head        ← nose (0)
    1: 11,  # l_shoulder  ← left_shoulder (11)
    2: 12,  # r_shoulder  ← right_shoulder (12)
    3: 13,  # l_elbow     ← left_elbow (13)
    4: 14,  # r_elbow     ← right_elbow (14)
    5: 15,  # l_wrist     ← left_wrist (15)
    6: 16,  # r_wrist     ← right_wrist (16)
    7: 23,  # l_hip       ← left_hip (23)
    8: 24,  # r_hip       ← right_hip (24)
    9: 25,  # l_knee      ← left_knee (25)
    10: 26, # r_knee      ← right_knee (26)
    11: 27, # l_ankle     ← left_ankle (27)
    12: 28, # r_ankle     ← right_ankle (28)
}

# Penn Action parent joint indices for bone vector computation
# -1 means the joint's reference is the torso center
PENN_JOINT_PARENTS = [-1, -1, -1, 1, 2, 3, 4, -1, -1, 7, 8, 9, 10]
# Torso joints used to compute the torso center
PENN_TORSO_JOINTS = [1, 2, 7, 8]  # l_shoulder, r_shoulder, l_hip, r_hip

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_TITLE = "Human Activity Tracker API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Predict human activities from smartphone sensor data (accelerometer + gyroscope) "
    "or from uploaded video files using deep learning models."
)

# Max video upload size in bytes (100 MB)
MAX_VIDEO_SIZE_BYTES: int = 100 * 1024 * 1024

# Allowed video MIME types
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/x-msvideo", "video/quicktime", "video/avi"}

