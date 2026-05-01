"""
CameraReader module for camera frame capture.
Probes native pixel format and returns camera frames with capture timestamps.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraReader:
    """Reads frames from a camera device."""

    def __init__(
        self,
        camera_id: int = 0,
        backend: int = cv2.CAP_ANY,
        camera_fourcc: str = "",
        camera_width: int = 0,
        camera_height: int = 0,
        camera_fps: int = 0,
    ):
        self._camera_id = camera_id
        self._backend = backend
        self._camera_fourcc = camera_fourcc
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_fps = camera_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self.pixel_format: str = "bgr"
        self.fps: float = 0.0

    def _probe_format(self, cap: cv2.VideoCapture, probe_frame: np.ndarray) -> str:
        if probe_frame is None:
            return "bgr"
        channels = probe_frame.shape[2] if probe_frame.ndim == 3 else 1
        if channels == 1:
            return "gray"
        cap_fmt = int(cap.get(cv2.CAP_PROP_FORMAT))
        if cap_fmt in (cv2.COLOR_YUV2RGB_YUY2, cv2.COLOR_YUV2BGR_YUY2, 16):
            return "yuyv"
        if channels == 3:
            return "rgb"
        return "bgr"

    def _open_camera(self) -> Tuple[cv2.VideoCapture, dict]:
        backends = {
            "auto": [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", None)],
            "msmf": [("msmf", cv2.CAP_MSMF), ("any", None)],
            "dshow": [("dshow", cv2.CAP_DSHOW), ("any", None)],
            "any": [("any", None)],
        }

        backend_strategy = "any"
        if self._backend == cv2.CAP_MSMF:
            backend_strategy = "msmf"
        elif self._backend == cv2.CAP_DSHOW:
            backend_strategy = "dshow"

        errors = []

        for name, backend_value in backends.get(backend_strategy, backends["auto"]):
            cap = (
                cv2.VideoCapture(self._camera_id, backend_value)
                if backend_value is not None
                else cv2.VideoCapture(self._camera_id)
            )
            if not cap.isOpened():
                cap.release()
                errors.append(f"{name}: open failed")
                continue

            fourcc = (self._camera_fourcc or "").strip().upper()
            if len(fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if self._camera_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._camera_width))
            if self._camera_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._camera_height))
            if self._camera_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(self._camera_fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

            ok, probe = cap.read()
            if not ok:
                cap.release()
                errors.append(f"{name}: read failed")
                continue

            pixel_format = self._probe_format(cap, probe)

            info = {
                "backend": name,
                "index": self._camera_id,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "pixel_format": pixel_format,
            }
            return cap, info

        raise RuntimeError(f"Unable to open webcam ({'; '.join(errors)})")

    def open(self) -> dict:
        """Open the camera and return info dict."""
        self._cap, info = self._open_camera()
        self.pixel_format = info["pixel_format"]
        self.fps = float(info.get("fps", 0.0) or 0.0)
        logger.info(
            f"CameraReader: Camera backend={info['backend']} index={info['index']} "
            f"{info['width']}x{info['height']} {info['fps']:.1f}fps pixel_format={self.pixel_format}",
        )
        logger.info("CameraReader: Webcam capture started.")
        return info

    def read_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """Read one frame from the camera and return it with a timestamp.

        Returns:
            Tuple of (frame, timestamp_ms). Frame is None on failure.
        """
        if self._cap is None:
            return None, 0

        ok, frame = self._cap.read()
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                logger.warning("CameraReader: cap.read() returned False")
            elif self._consecutive_failures % 100 == 0:
                logger.warning(
                    f"CameraReader: cap.read() has failed {self._consecutive_failures} consecutive times"
                )
            return None, 0

        if self._consecutive_failures > 0:
            logger.info(
                f"CameraReader: cap.read() recovered after {self._consecutive_failures} failures"
            )
            self._consecutive_failures = 0

        return frame, int(time.time() * 1000)

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("CameraReader: Released.")
