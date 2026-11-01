"""
test_backend.py
----------------
Automated test script for the Human Activity Tracker backend.
Tests:
  1. GET /health
  2. POST /sensor/predict (single 128-sample window)
  3. POST /sensor/push (incremental stream buffering & sliding window)
  4. WS /ws/stream/{device_id} (real-time WebSocket streaming)
"""

import asyncio
import json
import time
import urllib.request
import numpy as np

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/stream/simulated-phone"


def test_health():
    print("\n--- 1. Testing GET /health ---")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"Status: {data.get('status')}")
        print(f"Sensor Model: {data['models']['sensor']['name']} (Acc: {data['models']['sensor']['test_accuracy_pct']}%)")
        print(f"Video Model:  {data['models']['video']['name']} (Acc: {data['models']['video']['test_accuracy_pct']}%)")
        assert data["status"] == "ok"
    print("GET /health PASSED")


def test_sensor_predict():
    print("\n--- 2. Testing POST /sensor/predict ---")
    # Generate 128 dummy readings simulating standing (gravity ~ 9.8 on Z)
    window = {
        "total_acc_x": (np.random.randn(128) * 0.05).tolist(),
        "total_acc_y": (np.random.randn(128) * 0.05).tolist(),
        "total_acc_z": (9.8 + np.random.randn(128) * 0.05).tolist(),
        "gyro_x": (np.random.randn(128) * 0.01).tolist(),
        "gyro_y": (np.random.randn(128) * 0.01).tolist(),
        "gyro_z": (np.random.randn(128) * 0.01).tolist(),
    }
    payload = json.dumps({"device_id": "test-device-1", "window": window}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/sensor/predict",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
        print(f"Predicted Activity: {result['activity']}")
        print(f"Confidence:         {result['confidence'] * 100:.2f}%")
        print(f"Inference Latency:  {result['inference_ms']} ms")
        assert "activity" in result
    print("POST /sensor/predict PASSED")


def test_sensor_push():
    print("\n--- 3. Testing POST /sensor/push (incremental stream) ---")
    device_id = "test-device-push"
    # Push 60 samples -> Should report buffering
    for i in range(60):
        sample = {
            "device_id": device_id,
            "timestamp": time.time(),
            "total_acc_x": 0.05,
            "total_acc_y": 0.02,
            "total_acc_z": 9.81,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
        }
        req = urllib.request.Request(
            f"{BASE_URL}/sensor/push",
            data=json.dumps(sample).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
    
    print(f"After 60 samples: status = {data['status']}, buffer_size = {data['buffer_size']}")
    assert data["status"] == "buffering"

    # Push 68 more samples -> Total 128 -> Should trigger prediction!
    for i in range(68):
        sample = {
            "device_id": device_id,
            "timestamp": time.time(),
            "total_acc_x": 0.05,
            "total_acc_y": 0.02,
            "total_acc_z": 9.81,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
        }
        req = urllib.request.Request(
            f"{BASE_URL}/sensor/push",
            data=json.dumps(sample).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

    print(f"After 128 samples: status = {data['status']}, activity = {data.get('activity')}, conf = {data.get('confidence')}")
    assert data["status"] == "predicted"
    print("POST /sensor/push PASSED")


async def test_websocket_stream():
    print("\n--- 4. Testing WS /ws/stream/{device_id} (Real-time WebSocket) ---")
    try:
        import websockets
    except ImportError:
        print("websockets library not installed. Skipping live WS test.")
        return

    async with websockets.connect(WS_URL) as ws:
        print("WebSocket connected to server.")
        
        # Send 130 samples rapidly to trigger the first window prediction
        prediction_received = False
        for i in range(130):
            sample = {
                "timestamp": time.time(),
                "total_acc_x": float(np.random.normal(0, 0.1)),
                "total_acc_y": float(np.random.normal(0, 0.1)),
                "total_acc_z": float(9.8 + np.random.normal(0, 0.1)),
                "gyro_x": float(np.random.normal(0, 0.01)),
                "gyro_y": float(np.random.normal(0, 0.01)),
                "gyro_z": float(np.random.normal(0, 0.01)),
            }
            await ws.send(json.dumps(sample))
            resp = json.loads(await ws.recv())
            if resp.get("type") == "prediction":
                print(f"Server streamed prediction: {resp['activity']} ({resp['confidence']*100:.1f}%) in {resp['inference_ms']}ms")
                prediction_received = True
                break

        assert prediction_received, "Expected prediction message from WebSocket"
        print("WebSocket stream test PASSED")


if __name__ == "__main__":
    print("Starting backend tests against http://127.0.0.1:8000 ...")
    test_health()
    test_sensor_predict()
    test_sensor_push()
    asyncio.run(test_websocket_stream())
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

