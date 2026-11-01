"""
services/buffer.py
------------------
Thread-safe per-device rolling window buffer for real-time sensor streaming.

Each connected device (identified by device_id) gets its own buffer that
accumulates raw sensor samples. Once the buffer reaches SENSOR_WINDOW_SIZE
(128) samples, a prediction is triggered. The buffer then slides forward
by SENSOR_SLIDE_SIZE (64) samples (50 % overlap), so a new prediction is
made every ~1.28 seconds of continuous streaming.

All buffer operations are synchronous in-memory deque operations protected
by threading.Lock for thread safety across concurrent workers.
"""

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Optional

from backend.config import SENSOR_SLIDE_SIZE, SENSOR_WINDOW_SIZE


@dataclass
class DeviceBuffer:
    """Holds the sliding window state for one device."""
    samples: deque = field(
        default_factory=lambda: deque(maxlen=SENSOR_WINDOW_SIZE)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    samples_since_last_pred: int = 0


class SensorBuffer:
    """
    Registry of per-device buffers.
    """

    def __init__(self) -> None:
        self._devices: dict[str, DeviceBuffer] = {}
        self._registry_lock = threading.Lock()

    def get_or_create(self, device_id: str) -> DeviceBuffer:
        """Return existing buffer for device_id, creating one if needed."""
        with self._registry_lock:
            if device_id not in self._devices:
                self._devices[device_id] = DeviceBuffer()
            return self._devices[device_id]

    def push(
        self,
        device_id: str,
        sample: dict,
    ) -> tuple[bool, Optional[dict[str, list]]]:
        """
        Add one raw sample to the device buffer.

        Args:
            device_id: unique device/session identifier.
            sample: dict with keys:
                      total_acc_x, total_acc_y, total_acc_z,
                      gyro_x, gyro_y, gyro_z

        Returns:
            (ready, window_data) where:
              ready       — True when a full 128-sample window is available
              window_data — dict of 6 lists (128 values each) when ready,
                            else None.
        """
        device = self.get_or_create(device_id)

        with device.lock:
            device.samples.append(sample)
            device.samples_since_last_pred += 1

            buf_len = len(device.samples)

            # Not enough samples yet
            if buf_len < SENSOR_WINDOW_SIZE:
                return False, None

            # Buffer is full, check if we accumulated enough NEW samples since last prediction
            if device.samples_since_last_pred < SENSOR_SLIDE_SIZE:
                return False, None

            # Window is ready: extract the 128 most recent samples
            window_samples = list(device.samples)

            window_data: dict[str, list] = {
                "total_acc_x": [s["total_acc_x"] for s in window_samples],
                "total_acc_y": [s["total_acc_y"] for s in window_samples],
                "total_acc_z": [s["total_acc_z"] for s in window_samples],
                "gyro_x":      [s["gyro_x"]      for s in window_samples],
                "gyro_y":      [s["gyro_y"]      for s in window_samples],
                "gyro_z":      [s["gyro_z"]      for s in window_samples],
            }

            # Reset slide counter
            device.samples_since_last_pred = 0

            return True, window_data

    def clear(self, device_id: str) -> None:
        """Remove and reset a device buffer."""
        with self._registry_lock:
            if device_id in self._devices:
                del self._devices[device_id]

    def device_count(self) -> int:
        return len(self._devices)

    def buffer_size(self, device_id: str) -> int:
        """Return current number of buffered samples for a device (0 if unknown)."""
        buf = self._devices.get(device_id)
        return len(buf.samples) if buf else 0


# Module-level singleton — imported by routers
sensor_buffer = SensorBuffer()
