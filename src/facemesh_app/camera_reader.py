"""
CameraReader module for camera frame capture.

The capture pipeline doesn't benefit from high resolution / high fps because
mediapipe's FaceLandmarker downsamples to fixed sizes internally (128x128
detector, 256x256 landmarks). High-res capture only inflates `cap.read` and
`mp.Image` buffer copies. So instead of letting users configure a mode they
can't usefully exploit, we probe a curated ladder of (backend, fourcc, size)
candidates from cheapest to most expensive and accept the first one that
works on this machine.
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Candidate modes in order of preference.
#
# Why 1024x768 is the floor:
#  - MediaPipe's landmarks model has a fixed 256x256 input. With a face
#    occupying ~1/3 of frame height, 768 vertical pixels yields a face
#    crop of ~256 px native — the internal crop->256 resize is roughly
#    identity. Anything below 768 forces an *upsample* of the face
#    region, blurring landmarks for no CPU saving worth caring about.
#  - 1024x768 is also a clean integer multiple of 256.
#
# Why this order:
#  - We start at 1024x768 (the alignment sweet spot) and go *upward* if
#    the camera doesn't expose it — never below.
#  - For each size we try NV12 first; it's the camera-native chroma format
#    on modern UVC and lets MSMF's frame transformer scale in hardware.
#    YUY2 at the same size is the fallback fourcc.
#  - 4:3 modes are preferred over 16:9 at the same height because they
#    add headroom/chin pixels rather than side margins; the face fills
#    more of the frame.
#  - DShow is only used at native because asking DShow for a smaller
#    resolution triggers a CPU-side scaling slow path (~73 ms per cap.read
#    measured on our test camera).
#  - The timing check below catches any candidate that *looks* fine via
#    set/get round-trip but actually hits a slow path.
#
# Format: (backend_name, fourcc_or_None, width, height, fps_or_0, label)
_CANDIDATE_MODES: List[Tuple[str, Optional[str], int, int, int, str]] = [
    ("msmf",  "NV12", 1024,  768, 30, "MSMF NV12 1024x768"),
    ("msmf",  "YUY2", 1024,  768, 30, "MSMF YUY2 1024x768"),
    ("msmf",  "NV12", 1280,  960, 30, "MSMF NV12 1280x960"),
    ("msmf",  "YUY2", 1280,  960, 30, "MSMF YUY2 1280x960"),
    ("msmf",  "NV12", 1920, 1080, 30, "MSMF NV12 1920x1080"),
    ("msmf",  "NV12", 2560, 1440, 30, "MSMF NV12 2560x1440"),
    ("msmf",  None,      0,    0,  0, "MSMF native"),
    ("dshow", None,      0,    0,  0, "DShow native"),
    ("any",   None,      0,    0,  0, "any backend native"),
]

_BACKEND_MAP = {
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any":   None,
}

# Reject a candidate whose median read wall time exceeds this. At 30 fps the
# camera period is ~33 ms; a healthy read sits at or below that. The DShow
# CPU-scaling trap manifests as ~70 ms+, so 60 ms cleanly separates the two.
_MAX_READ_MS = 60.0


class CameraReader:
    """Reads frames from a camera device, picking the most CPU-efficient mode."""

    def __init__(self, camera_id: int = 0):
        self._camera_id = camera_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self.pixel_format: str = "bgr"
        self.fps: float = 0.0
        # Width/height we actually negotiated, used to reshape flat NV12
        # buffers that MSMF delivers as `(1, w*h*3/2)` raw byte runs.
        self._frame_width: int = 0
        self._frame_height: int = 0

    def _probe_format(self, cap: cv2.VideoCapture, probe_frame: np.ndarray) -> str:
        # Identify the buffer layout we actually got back. Channel count alone
        # is ambiguous (NV12 arrives as a single-channel `(h*1.5, w)` buffer
        # that looks identical to grayscale), so consult the fourcc first.
        if probe_frame is None:
            return "bgr"

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = ""
        if fourcc_int > 0:
            fourcc = "".join(
                chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
            ).strip().strip("\x00").upper()

        channels = probe_frame.shape[2] if probe_frame.ndim == 3 else 1

        if channels == 1:
            if fourcc == "NV12":
                return "nv12"
            if fourcc in ("YUY2", "YUYV"):
                return "yuyv"
            return "gray"

        # 3-channel buffer: OpenCV's capture backends always deliver BGR by
        # default (regardless of CAP_PROP_CONVERT_RGB). Treating it as RGB
        # silently swaps red/blue going into the model.
        return "bgr"

    def _try_candidate(
        self,
        backend_name: str,
        fourcc: Optional[str],
        width: int,
        height: int,
        fps: int,
        label: str,
    ) -> Optional[Tuple[cv2.VideoCapture, dict]]:
        """Open a candidate, validate the negotiated mode, time some reads.

        Returns the (cap, info) pair on success, None if this candidate is
        rejected for any reason (open failed, dims didn't match, reads
        too slow, etc.).
        """
        backend_const = _BACKEND_MAP.get(backend_name, None)
        cap = (
            cv2.VideoCapture(self._camera_id, backend_const)
            if backend_const is not None
            else cv2.VideoCapture(self._camera_id)
        )
        if not cap.isOpened():
            cap.release()
            logger.info("CameraReader: %s — open failed", label)
            return None

        if fourcc and len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        ok, probe = cap.read()
        if not ok:
            cap.release()
            logger.info("CameraReader: %s — first read failed", label)
            return None

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If we asked for a specific size and the driver coerced us to
        # something else, don't accept — the candidate ladder has a "native"
        # entry below that handles the no-resize case explicitly.
        if width > 0 and height > 0 and (actual_w != width or actual_h != height):
            cap.release()
            logger.info(
                "CameraReader: %s — dims rejected (got %dx%d, wanted %dx%d)",
                label, actual_w, actual_h, width, height,
            )
            return None

        # Time a handful of reads. The first one warming up the driver may
        # be slow, so discard it; measure the next four.
        timings: List[float] = []
        for _ in range(4):
            t0 = time.perf_counter()
            ok, _ = cap.read()
            if not ok:
                cap.release()
                logger.info("CameraReader: %s — sustained read failed", label)
                return None
            timings.append(time.perf_counter() - t0)
        timings.sort()
        median_ms = timings[len(timings) // 2] * 1000.0
        if median_ms > _MAX_READ_MS:
            cap.release()
            logger.info(
                "CameraReader: %s — median read %.0f ms exceeds %.0f ms cap "
                "(likely driver CPU-scaling slow path)",
                label, median_ms, _MAX_READ_MS,
            )
            return None

        pixel_format = self._probe_format(cap, probe)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        )

        info = {
            "backend": backend_name,
            "candidate": label,
            "index": self._camera_id,
            "width": actual_w,
            "height": actual_h,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": fourcc_str,
            "pixel_format": pixel_format,
            "median_read_ms": median_ms,
        }
        return cap, info

    def _open_camera(self) -> Tuple[cv2.VideoCapture, dict]:
        """Walk the candidate ladder, accept the first that works."""
        for cand in _CANDIDATE_MODES:
            result = self._try_candidate(*cand)
            if result is not None:
                return result
        raise RuntimeError(
            "Unable to open camera with any of the candidate modes"
        )

    def open(self) -> dict:
        """Open the camera and return info dict."""
        self._cap, info = self._open_camera()
        self.pixel_format = info["pixel_format"]
        self.fps = float(info.get("fps", 0.0) or 0.0)
        self._frame_width = int(info.get("width", 0) or 0)
        self._frame_height = int(info.get("height", 0) or 0)
        logger.info(
            f"CameraReader: accepted '{info.get('candidate', '?')}' — "
            f"backend={info['backend']} index={info['index']} "
            f"{info['width']}x{info['height']} {info['fps']:.1f}fps "
            f"fourcc={info.get('fourcc', '????')!r} pixel_format={self.pixel_format} "
            f"median_read={info.get('median_read_ms', 0.0):.1f}ms",
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

        if (
            self.pixel_format == "nv12"
            and self._frame_height > 0
            and self._frame_width > 0
            and frame is not None
        ):
            # MSMF delivers NV12 as a flat `(1, w*h*3/2)` byte buffer; cv2's
            # NV12 converters need it as `(h*3/2, w)`. Reshape is a view —
            # zero copy — when the buffer is already contiguous.
            expected = self._frame_height * self._frame_width * 3 // 2
            if frame.size == expected:
                frame = frame.reshape(self._frame_height * 3 // 2, self._frame_width)

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
