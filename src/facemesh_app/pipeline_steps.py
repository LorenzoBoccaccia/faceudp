"""
Pipeline steps for face processing.
Each step processes data and passes it to the next step in the pipeline.
"""

import logging
import socket
import struct
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from .calibration import CalibratedFaceAndGazeEvent
from .facemesh_dao import FaceMeshEvent
from .gaze_primitives import collect_gaze_primitives, draw_gaze_primitives_cv2

logger = logging.getLogger(__name__)


class FaceMeshStep:
    """First pipeline step: Extract face mesh data from frames using MediaPipe FaceLandmarker."""

    _CONVERT_MAP = {
        "bgr": cv2.COLOR_BGR2RGB,
        "yuyv": cv2.COLOR_YUV2RGB_YUY2,
        "nv12": cv2.COLOR_YUV2RGB_NV12,
    }

    def __init__(self, face_landmarker: vision.FaceLandmarker):
        self.face_landmarker = face_landmarker
        self._last_timestamp_ms = -1

    def receive_frame(
        self, frame, timestamp_ms: int, pixel_format: str = "bgr"
    ) -> Optional[FaceMeshEvent]:
        if frame is None:
            logger.warning("Received None frame in FaceMeshStep")
            return None

        try:
            if pixel_format == "rgb":
                frame_rgb = frame
            else:
                code = self._CONVERT_MAP.get(pixel_format, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.cvtColor(frame, code)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int(timestamp_ms)
            if ts <= self._last_timestamp_ms:
                ts = self._last_timestamp_ms + 1
            self._last_timestamp_ms = ts

            result = self.face_landmarker.detect_for_video(mp_image, ts)
            evt = FaceMeshEvent.from_landmarker_result(result, ts=ts)
            return evt

        except Exception as e:
            logger.error(f"Error processing frame in FaceMeshStep: {e}")
            return None


class CalibrationAdapterStep:
    """Second pipeline step: Convert FaceMeshEvent to CalibratedFaceAndGazeEvent.

    This adapter step bridges the face mesh data with calibration and display geometry
    to create the comprehensive event used by downstream steps.
    """

    def __init__(
        self,
        pitch_calibration: float = 0.0,
        yaw_calibration: float = 0.0,
        roll_calibration: float = 0.0,
        face_center_yaw: float = 0.0,
        face_center_pitch: float = 0.0,
        center_zeta: float = 1200.0,
        yaw_coefficient_positive: float = 1.0,
        yaw_coefficient_negative: float = 1.0,
        pitch_coefficient_positive: float = 1.0,
        pitch_coefficient_negative: float = 1.0,
        yaw_from_pitch_coupling: float = 0.0,
        pitch_from_yaw_coupling: float = 0.0,
        eye_yaw_min: float = -1.0,
        eye_yaw_max: float = 1.0,
        eye_pitch_min: float = -1.0,
        eye_pitch_max: float = 1.0,
        face_center_x: float = 0.0,
        face_center_y: float = 0.0,
        face_center_z: float = 1200.0,
        screen_center_cam_x: float = 0.0,
        screen_center_cam_y: float = 0.0,
        screen_center_cam_z: float = 1200.0,
        screen_axis_x_x: float = 1.0,
        screen_axis_x_y: float = 0.0,
        screen_axis_x_z: float = 0.0,
        screen_axis_y_x: float = 0.0,
        screen_axis_y_y: float = 1.0,
        screen_axis_y_z: float = 0.0,
        screen_scale_x: float = 1.0,
        screen_scale_y: float = 1.0,
        screen_fit_rmse: float = -1.0,
        display_width: int = 1920,
        display_height: int = 1080,
        origin_x: float = 960.0,
        origin_y: float = 540.0,
    ):
        """Initialize CalibrationAdapterStep.

        Args:
            pitch_calibration: Pitch calibration value in degrees
            yaw_calibration: Yaw calibration value in degrees
            roll_calibration: Roll calibration value in degrees
            display_width: Display width in pixels
            display_height: Display height in pixels
            origin_x: X coordinate of origin (e.g., screen center)
            origin_y: Y coordinate of origin
        """
        self.pitch_calibration = float(pitch_calibration)
        self.yaw_calibration = float(yaw_calibration)
        self.roll_calibration = float(roll_calibration)
        self.face_center_yaw = float(face_center_yaw)
        self.face_center_pitch = float(face_center_pitch)
        self.center_zeta = float(center_zeta)
        self.yaw_coefficient_positive = float(yaw_coefficient_positive)
        self.yaw_coefficient_negative = float(yaw_coefficient_negative)
        self.pitch_coefficient_positive = float(pitch_coefficient_positive)
        self.pitch_coefficient_negative = float(pitch_coefficient_negative)
        self.yaw_from_pitch_coupling = float(yaw_from_pitch_coupling)
        self.pitch_from_yaw_coupling = float(pitch_from_yaw_coupling)
        self.eye_yaw_min = float(eye_yaw_min)
        self.eye_yaw_max = float(eye_yaw_max)
        self.eye_pitch_min = float(eye_pitch_min)
        self.eye_pitch_max = float(eye_pitch_max)
        self.face_center_x = float(face_center_x)
        self.face_center_y = float(face_center_y)
        self.face_center_z = float(face_center_z)
        self.screen_center_cam_x = float(screen_center_cam_x)
        self.screen_center_cam_y = float(screen_center_cam_y)
        self.screen_center_cam_z = float(screen_center_cam_z)
        self.screen_axis_x_x = float(screen_axis_x_x)
        self.screen_axis_x_y = float(screen_axis_x_y)
        self.screen_axis_x_z = float(screen_axis_x_z)
        self.screen_axis_y_x = float(screen_axis_y_x)
        self.screen_axis_y_y = float(screen_axis_y_y)
        self.screen_axis_y_z = float(screen_axis_y_z)
        self.screen_scale_x = float(screen_scale_x)
        self.screen_scale_y = float(screen_scale_y)
        self.screen_fit_rmse = float(screen_fit_rmse)
        self.display_width = int(display_width)
        self.display_height = int(display_height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

    def update_calibration(
        self,
        pitch: float,
        yaw: float,
        roll: float,
        *,
        face_center_yaw: Optional[float] = None,
        face_center_pitch: Optional[float] = None,
        center_zeta: Optional[float] = None,
        yaw_coefficient_positive: Optional[float] = None,
        yaw_coefficient_negative: Optional[float] = None,
        pitch_coefficient_positive: Optional[float] = None,
        pitch_coefficient_negative: Optional[float] = None,
        yaw_from_pitch_coupling: Optional[float] = None,
        pitch_from_yaw_coupling: Optional[float] = None,
        eye_yaw_min: Optional[float] = None,
        eye_yaw_max: Optional[float] = None,
        eye_pitch_min: Optional[float] = None,
        eye_pitch_max: Optional[float] = None,
        face_center_x: Optional[float] = None,
        face_center_y: Optional[float] = None,
        face_center_z: Optional[float] = None,
        screen_center_cam_x: Optional[float] = None,
        screen_center_cam_y: Optional[float] = None,
        screen_center_cam_z: Optional[float] = None,
        screen_axis_x_x: Optional[float] = None,
        screen_axis_x_y: Optional[float] = None,
        screen_axis_x_z: Optional[float] = None,
        screen_axis_y_x: Optional[float] = None,
        screen_axis_y_y: Optional[float] = None,
        screen_axis_y_z: Optional[float] = None,
        screen_scale_x: Optional[float] = None,
        screen_scale_y: Optional[float] = None,
        screen_fit_rmse: Optional[float] = None,
    ) -> None:
        """Update calibration values.

        Args:
            pitch: New pitch calibration value in degrees
            yaw: New yaw calibration value in degrees
            roll: New roll calibration value in degrees
        """
        self.pitch_calibration = float(pitch)
        self.yaw_calibration = float(yaw)
        self.roll_calibration = float(roll)
        if face_center_yaw is not None:
            self.face_center_yaw = float(face_center_yaw)
        if face_center_pitch is not None:
            self.face_center_pitch = float(face_center_pitch)
        if center_zeta is not None:
            self.center_zeta = float(center_zeta)
        if yaw_coefficient_positive is not None:
            self.yaw_coefficient_positive = float(yaw_coefficient_positive)
        if yaw_coefficient_negative is not None:
            self.yaw_coefficient_negative = float(yaw_coefficient_negative)
        if pitch_coefficient_positive is not None:
            self.pitch_coefficient_positive = float(pitch_coefficient_positive)
        if pitch_coefficient_negative is not None:
            self.pitch_coefficient_negative = float(pitch_coefficient_negative)
        if yaw_from_pitch_coupling is not None:
            self.yaw_from_pitch_coupling = float(yaw_from_pitch_coupling)
        if pitch_from_yaw_coupling is not None:
            self.pitch_from_yaw_coupling = float(pitch_from_yaw_coupling)
        if eye_yaw_min is not None:
            self.eye_yaw_min = float(eye_yaw_min)
        if eye_yaw_max is not None:
            self.eye_yaw_max = float(eye_yaw_max)
        if eye_pitch_min is not None:
            self.eye_pitch_min = float(eye_pitch_min)
        if eye_pitch_max is not None:
            self.eye_pitch_max = float(eye_pitch_max)
        if face_center_x is not None:
            self.face_center_x = float(face_center_x)
        if face_center_y is not None:
            self.face_center_y = float(face_center_y)
        if face_center_z is not None:
            self.face_center_z = float(face_center_z)
        if screen_center_cam_x is not None:
            self.screen_center_cam_x = float(screen_center_cam_x)
        if screen_center_cam_y is not None:
            self.screen_center_cam_y = float(screen_center_cam_y)
        if screen_center_cam_z is not None:
            self.screen_center_cam_z = float(screen_center_cam_z)
        if screen_axis_x_x is not None:
            self.screen_axis_x_x = float(screen_axis_x_x)
        if screen_axis_x_y is not None:
            self.screen_axis_x_y = float(screen_axis_x_y)
        if screen_axis_x_z is not None:
            self.screen_axis_x_z = float(screen_axis_x_z)
        if screen_axis_y_x is not None:
            self.screen_axis_y_x = float(screen_axis_y_x)
        if screen_axis_y_y is not None:
            self.screen_axis_y_y = float(screen_axis_y_y)
        if screen_axis_y_z is not None:
            self.screen_axis_y_z = float(screen_axis_y_z)
        if screen_scale_x is not None:
            self.screen_scale_x = float(screen_scale_x)
        if screen_scale_y is not None:
            self.screen_scale_y = float(screen_scale_y)
        if screen_fit_rmse is not None:
            self.screen_fit_rmse = float(screen_fit_rmse)
        logger.debug(f"Calibration updated: pitch={pitch}, yaw={yaw}, roll={roll}")

    def update_display_geometry(
        self, width: int, height: int, origin_x: float, origin_y: float
    ) -> None:
        """Update display geometry.

        Args:
            width: Display width in pixels
            height: Display height in pixels
            origin_x: X coordinate of origin
            origin_y: Y coordinate of origin
        """
        self.display_width = int(width)
        self.display_height = int(height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        logger.debug(
            f"Display geometry updated: {width}x{height}, origin=({origin_x}, {origin_y})"
        )

    def receive_frame(
        self, frame: np.ndarray, face_mesh_event: Optional[FaceMeshEvent]
    ) -> Optional[CalibratedFaceAndGazeEvent]:
        """Create CalibratedFaceAndGazeEvent from FaceMeshEvent.

        This method combines the face mesh data with calibration and display geometry
        to create a comprehensive event for downstream processing.

        Args:
            frame: Input frame (may be needed for future extensions)
            face_mesh_event: Face mesh data from FaceMeshStep

        Returns:
            CalibratedFaceAndGazeEvent if face_mesh_event is not None, None otherwise
        """
        if face_mesh_event is None:
            logger.debug("FaceMeshEvent is None, returning None")
            return None
        if not face_mesh_event.has_face:
            return None
        if (
            face_mesh_event.combined_eye_gaze_yaw is None
            or face_mesh_event.combined_eye_gaze_pitch is None
            or face_mesh_event.head_yaw is None
            or face_mesh_event.head_pitch is None
            or face_mesh_event.camera_x is None
            or face_mesh_event.camera_y is None
            or face_mesh_event.camera_z is None
        ):
            return None

        try:
            calibrated_event = CalibratedFaceAndGazeEvent(
                face_mesh_event=face_mesh_event,
                pitch_calibration=self.pitch_calibration,
                yaw_calibration=self.yaw_calibration,
                roll_calibration=self.roll_calibration,
                face_center_yaw=self.face_center_yaw,
                face_center_pitch=self.face_center_pitch,
                center_zeta=self.center_zeta,
                yaw_coefficient_positive=self.yaw_coefficient_positive,
                yaw_coefficient_negative=self.yaw_coefficient_negative,
                pitch_coefficient_positive=self.pitch_coefficient_positive,
                pitch_coefficient_negative=self.pitch_coefficient_negative,
                yaw_from_pitch_coupling=self.yaw_from_pitch_coupling,
                pitch_from_yaw_coupling=self.pitch_from_yaw_coupling,
                eye_yaw_min=self.eye_yaw_min,
                eye_yaw_max=self.eye_yaw_max,
                eye_pitch_min=self.eye_pitch_min,
                eye_pitch_max=self.eye_pitch_max,
                face_center_x=self.face_center_x,
                face_center_y=self.face_center_y,
                face_center_z=self.face_center_z,
                screen_center_cam_x=self.screen_center_cam_x,
                screen_center_cam_y=self.screen_center_cam_y,
                screen_center_cam_z=self.screen_center_cam_z,
                screen_axis_x_x=self.screen_axis_x_x,
                screen_axis_x_y=self.screen_axis_x_y,
                screen_axis_x_z=self.screen_axis_x_z,
                screen_axis_y_x=self.screen_axis_y_x,
                screen_axis_y_y=self.screen_axis_y_y,
                screen_axis_y_z=self.screen_axis_y_z,
                screen_scale_x=self.screen_scale_x,
                screen_scale_y=self.screen_scale_y,
                screen_fit_rmse=self.screen_fit_rmse,
                display_width=self.display_width,
                display_height=self.display_height,
                origin_x=self.origin_x,
                origin_y=self.origin_y,
            )

            if face_mesh_event.has_face:
                logger.debug("CalibratedFaceAndGazeEvent created with face data")
            else:
                logger.debug("CalibratedFaceAndGazeEvent created without face data")

            return calibrated_event

        except Exception as e:
            logger.error(f"Error creating CalibratedFaceAndGazeEvent: {e}")
            return None


class CaptureStep:
    """Fourth pipeline step: Handle live preview display and frame counting.

    This step displays processed frames with overlays when enabled.
    It tracks frame count for statistics and handles keyboard input for quitting.
    Note: Actual capture saving is handled via callback mechanism in FrameDispatcher.
    """

    def __init__(self, enabled: bool = True):
        """Initialize capture step.

        Args:
            enabled: Whether capture step is active
        """
        self.enabled = enabled
        self.frame_count = 0

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable capture step.

        Args:
            enabled: Whether to enable the capture step
        """
        self.enabled = enabled

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
    ) -> None:
        """Process frame for capture/live preview.

        Args:
            frame: Input frame
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)

        Note:
            This method doesn't return anything - it handles display internally.
            Capture saving is handled via callback mechanism.
        """
        if not self.enabled:
            return

        if frame is None:
            logger.warning("Received None frame in CaptureStep")
            return

        # Increment frame count
        self.frame_count += 1

        # Display frame with overlays
        cv2.imshow("FaceMesh Live Preview", frame)

        # Handle keyboard input (ESC to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            logger.info("ESC pressed - capture step will exit")
            # Note: Actual exit handling should be done by the application loop
            # This step just processes the key press

    def get_frame_count(self) -> int:
        """Get total number of frames processed.

        Returns:
            Total frame count
        """
        return self.frame_count


class OverlayStep:
    """Fifth pipeline step: Render gaze dot and HUD overlay on frames.

    This step provides visual feedback by rendering a gaze dot showing the
    calibrated gaze position and optional HUD information with calibration status
    and head pose values.
    """

    # Constants for rendering
    DOT_RADIUS = 14
    SCALE = 14.0  # Pixels per degree
    BLUE = (70, 180, 255)  # Gaze dot color (same as overlay.py)
    WHITE = (255, 255, 255)
    HUD_BG = (20, 20, 20)
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1

    def __init__(self, enabled: bool = True, show_hud: bool = True):
        """Initialize overlay step.

        Args:
            enabled: Whether overlay rendering is active
            show_hud: Whether to show HUD information
        """
        self.enabled = enabled
        self.show_hud = show_hud

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable overlay rendering.

        Args:
            enabled: Whether to enable overlay rendering
        """
        self.enabled = enabled
        logger.debug(f"OverlayStep enabled: {enabled}")

    def set_show_hud(self, show_hud: bool) -> None:
        """Enable or disable HUD display.

        Args:
            show_hud: Whether to show HUD information
        """
        self.show_hud = show_hud
        logger.debug(f"OverlayStep show_hud: {show_hud}")

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
    ) -> np.ndarray:
        """Render overlay on frame.

        Args:
            frame: Input frame
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)

        Returns:
            Frame with overlay rendered (original frame if disabled)
        """
        if not self.enabled or calibrated_event is None:
            return frame

        if frame is None:
            logger.warning("Received None frame in OverlayStep")
            return None

        try:
            face_event = calibrated_event.face_mesh_event

            frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            runtime_evt = {
                "head_yaw": face_event.head_yaw if face_event else None,
                "head_pitch": face_event.head_pitch if face_event else None,
                "head_x": face_event.camera_x if face_event else None,
                "head_y": face_event.camera_y if face_event else None,
                "head_z": face_event.camera_z if face_event else None,
                "raw_combined_eye_gaze_yaw": (
                    face_event.combined_eye_gaze_yaw if face_event else None
                ),
                "raw_combined_eye_gaze_pitch": (
                    face_event.combined_eye_gaze_pitch if face_event else None
                ),
                "face_delta_yaw": calibrated_event.face_delta_yaw,
                "face_delta_pitch": calibrated_event.face_delta_pitch,
                "corrected_eye_yaw": calibrated_event.corrected_eye_yaw,
                "corrected_eye_pitch": calibrated_event.corrected_eye_pitch,
                "corrected_yaw": calibrated_event.corrected_yaw,
                "corrected_pitch": calibrated_event.corrected_pitch,
                "corrected_yaw_linear": calibrated_event.corrected_yaw_linear,
                "corrected_pitch_linear": calibrated_event.corrected_pitch_linear,
                "corrected_screen_x": calibrated_event.corrected_screen_x,
                "corrected_screen_y": calibrated_event.corrected_screen_y,
                "origin_x": calibrated_event.origin_x,
                "origin_y": calibrated_event.origin_y,
                "center_zeta": calibrated_event.center_zeta,
                "screen_center_cam_x": calibrated_event.screen_center_cam_x,
                "screen_center_cam_y": calibrated_event.screen_center_cam_y,
                "screen_center_cam_z": calibrated_event.screen_center_cam_z,
                "screen_axis_x_x": calibrated_event.screen_axis_x_x,
                "screen_axis_x_y": calibrated_event.screen_axis_x_y,
                "screen_axis_x_z": calibrated_event.screen_axis_x_z,
                "screen_axis_y_x": calibrated_event.screen_axis_y_x,
                "screen_axis_y_y": calibrated_event.screen_axis_y_y,
                "screen_axis_y_z": calibrated_event.screen_axis_y_z,
                "screen_scale_x": calibrated_event.screen_scale_x,
                "screen_scale_y": calibrated_event.screen_scale_y,
                "screen_fit_rmse": calibrated_event.screen_fit_rmse,
            }
            primitives = collect_gaze_primitives(
                runtime_evt,
                frame_width,
                frame_height,
                origin_x=calibrated_event.origin_x,
                origin_y=calibrated_event.origin_y,
            )
            draw_gaze_primitives_cv2(
                frame,
                primitives,
                radius=self.DOT_RADIUS,
                outline_thickness=2,
            )

            if self.show_hud:
                self._draw_hud(frame, face_event, calibrated_event)

            return frame

        except Exception as e:
            logger.warning(f"Error rendering overlay: {e}", exc_info=True)
            return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        face_event: Optional[FaceMeshEvent],
        calibrated_event: CalibratedFaceAndGazeEvent,
    ) -> None:
        """Draw HUD with calibration status and head pose values.

        Args:
            frame: Frame to draw on
            face_event: Face mesh event
            calibrated_event: Calibrated event with calibration values
        """
        frame_height, frame_width = frame.shape[:2]

        has_face = face_event.has_face if face_event else False
        head_yaw = (
            face_event.head_yaw
            if face_event and face_event.head_yaw is not None
            else 0.0
        )
        head_pitch = (
            face_event.head_pitch
            if face_event and face_event.head_pitch is not None
            else 0.0
        )
        roll = face_event.roll if face_event and face_event.roll is not None else 0.0
        corrected_yaw = calibrated_event.corrected_yaw
        corrected_pitch = calibrated_event.corrected_pitch

        lines = [
            f"FACE: {'YES' if has_face else 'NO'}",
            f"Head Pitch: {head_pitch:.1f}",
            f"Head Yaw: {head_yaw:.1f}",
            f"Corrected Pitch: {corrected_pitch:.1f}",
            f"Corrected Yaw: {corrected_yaw:.1f}",
            f"Roll: {roll:.1f}",
        ]

        # Calculate HUD box dimensions
        line_height = 20
        padding = 8
        max_text_width = max([len(line) * 10 for line in lines])  # Approximate
        box_width = max_text_width + 2 * padding
        box_height = len(lines) * line_height + 2 * padding

        # Position HUD in top-left corner with margin
        margin = 10
        box_x = margin
        box_y = margin

        # Ensure HUD fits within frame
        if box_x + box_width > frame_width:
            box_x = frame_width - box_width - margin
        if box_y + box_height > frame_height:
            box_y = frame_height - box_height - margin

        # Draw HUD background
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            self.HUD_BG,
            -1,
        )
        cv2.rectangle(
            frame, (box_x, box_y), (box_x + box_width, box_y + box_height), self.BLUE, 1
        )

        # Draw text
        for i, line in enumerate(lines):
            text_y = box_y + padding + (i + 1) * line_height
            cv2.putText(
                frame,
                line,
                (box_x + padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.FONT_SCALE,
                self.WHITE,
                self.FONT_THICKNESS,
            )


class UDPForwardStep:
    """Final pipeline step: Forward calibrated face and gaze data via UDP to external applications.

    This step sends the processed and calibrated data to external applications via UDP protocol.
    It is disabled by default and can be enabled when needed for real-time data streaming.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 4242, enabled: bool = False
    ):
        """Initialize UDP forward step.

        Args:
            host: Target host address (default: "127.0.0.1")
            port: Target port number (default: 4242)
            enabled: Whether UDP forwarding is active (default: False)
        """
        self.host = host
        self.port = port
        self.enabled = enabled
        self._socket = None

        # Initialize socket if enabled
        if self.enabled:
            self._create_socket()

        logger.debug(
            f"UDPForwardStep initialized: host={host}, port={port}, enabled={enabled}"
        )

    def _create_socket(self) -> None:
        """Create UDP socket for sending data."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)  # Non-blocking mode
            logger.debug(f"UDP socket created for {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Failed to create UDP socket: {e}")
            self._socket = None

    def _close_socket(self) -> None:
        """Close UDP socket."""
        if self._socket is not None:
            try:
                self._socket.close()
                logger.debug("UDP socket closed")
            except Exception as e:
                logger.warning(f"Error closing UDP socket: {e}")
            finally:
                self._socket = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable UDP forwarding.

        Args:
            enabled: Whether to enable UDP forwarding
        """
        if self.enabled == enabled:
            return

        self.enabled = enabled

        if enabled:
            self._create_socket()
        else:
            self._close_socket()

        logger.debug(f"UDPForwardStep enabled: {enabled}")

    def _serialize_event(self, event: CalibratedFaceAndGazeEvent) -> bytes:
        """Serialize calibrated event to OpenTrack UDP payload.

        Args:
            event: Calibrated face and gaze event

        Returns:
            Binary OpenTrack pose payload
        """
        face_event = event.face_mesh_event

        head_x = (
            float(face_event.camera_x)
            if face_event and face_event.camera_x is not None
            else 0.0
        )
        head_y = (
            float(face_event.camera_y)
            if face_event and face_event.camera_y is not None
            else 0.0
        )
        head_z = (
            float(face_event.camera_z)
            if face_event and face_event.camera_z is not None
            else 0.0
        )
        yaw = float(event.corrected_yaw)
        # Facemesh derives pitch in an OpenCV +y-DOWN frame where +pitch means
        # user looking UP. Opentrack's world is +y-UP and its default mapping
        # treats +pitch as nose-DOWN, so flip the sign at the boundary.
        pitch = -float(event.corrected_pitch)
        roll = (
            float(face_event.roll)
            if face_event and face_event.roll is not None
            else 0.0
        )
        return struct.pack("<6d", head_x, head_y, head_z, yaw, pitch, roll)

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
    ) -> None:
        """Forward calibrated data via UDP.

        Args:
            frame: Input frame (not used but kept for interface consistency)
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)

        Note:
            This method doesn't return anything - it sends data via UDP.
        """
        if not self.enabled:
            return

        if calibrated_event is None:
            logger.debug("Calibrated event is None, skipping UDP forward")
            return

        if self._socket is None:
            logger.warning("UDP socket is None, skipping UDP forward")
            return

        try:
            message_bytes = self._serialize_event(calibrated_event)

            self._socket.sendto(message_bytes, (self.host, self.port))

            logger.debug(
                f"UDP message sent to {self.host}:{self.port}: {len(message_bytes)} bytes"
            )

        except (socket.error, OSError) as e:
            logger.warning(f"Socket error sending UDP message: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error sending UDP message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending UDP message: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self._close_socket()
