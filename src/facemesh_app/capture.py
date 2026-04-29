"""
Capture module for FaceMesh application.
Handles mesh data capture, screenshot generation, and test data saving.
"""

import json
import math
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from .facemesh_dao import (
    FaceMeshEvent,
    safe_float,
    clamp,
    LEFT_IRIS_CENTER_IDX,
    RIGHT_IRIS_CENTER_IDX,
    LEFT_IRIS_RING_IDXS,
    RIGHT_IRIS_RING_IDXS,
    LEFT_EYE_INNER_IDX,
    LEFT_EYE_OUTER_IDX,
    RIGHT_EYE_INNER_IDX,
    RIGHT_EYE_OUTER_IDX,
    NOSE_BRIDGE_IDX,
    NOSE_BASE_IDX,
)


# Constants
CAPTURE_DIR = Path("captures")
FACE_MESH_CONNECTIONS = list(vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)

# Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (80, 230, 120)
ORANGE = (0, 165, 255)
CYAN = (255, 220, 40)
MAGENTA = (255, 80, 255)
YELLOW = (0, 255, 255)
HUD_BG = (20, 20, 20)
HUD_BORDER = (230, 230, 230)
HUD_TEXT = (245, 245, 245)


def ms_now():
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def _lm_to_px(lm, w, h, mirror_x: bool = False):
    """Convert landmark to pixel coordinates."""
    if hasattr(lm, "x") and hasattr(lm, "y"):
        lx = safe_float(getattr(lm, "x", 0.0), 0.0)
        ly = safe_float(getattr(lm, "y", 0.0), 0.0)
    elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
        lx = safe_float(lm[0], 0.0)
        ly = safe_float(lm[1], 0.0)
    else:
        lx, ly = 0.0, 0.0
    if mirror_x:
        lx = 1.0 - lx
    x = int(clamp(round(lx * w), 0, w - 1))
    y = int(clamp(round(ly * h), 0, h - 1))
    return x, y


def _fmt_num(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    v = safe_float(value, float("nan"))
    if v != v:  # NaN
        return "n/a"
    return f"{v:.{precision}f}"


def _safe_lm_xy(lm) -> Optional[Tuple[float, float]]:
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return safe_float(getattr(lm, "x", 0.0), 0.0), safe_float(
            getattr(lm, "y", 0.0), 0.0
        )
    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        return safe_float(lm[0], 0.0), safe_float(lm[1], 0.0)
    return None


def _lm_points_px(
    landmarks: List[Any], indices: tuple[int, ...], w: int, h: int, mirror_x: bool
) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for idx in indices:
        if 0 <= int(idx) < len(landmarks):
            points.append(_lm_to_px(landmarks[int(idx)], w, h, mirror_x=mirror_x))
    return points


def _center_px(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not points:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = max(1, len(points))
    return int(round(sx / n)), int(round(sy / n))


def _build_event_lines(snap_evt: Any, landmarks: List[Any]) -> List[str]:
    lines: List[str] = []
    if isinstance(snap_evt, FaceMeshEvent):
        lines.append(f"type={snap_evt.type} ts={snap_evt.ts}")
        lines.append(f"face={snap_evt.has_face} landmarks={snap_evt.landmark_count}")
        lines.append(
            "head y/p/r="
            f"{_fmt_num(snap_evt.head_yaw, 2)}/{_fmt_num(snap_evt.head_pitch, 2)}/{_fmt_num(snap_evt.roll, 2)}"
        )
        lines.append(
            "tx/ty/tz="
            f"{_fmt_num(snap_evt.x, 4)}/{_fmt_num(snap_evt.y, 4)}/{_fmt_num(snap_evt.z, 4)}"
        )

        mask_meta = snap_evt.face_mask_segment_meta()
        if mask_meta:
            mask_shape = mask_meta.get("shape", "n/a")
            mask_dtype = mask_meta.get("dtype", "n/a")
            lines.append(
                f"mask type={mask_meta.get('type', 'n/a')} shape={mask_shape} dtype={mask_dtype}"
            )
        else:
            lines.append("mask: none")

        blendshape_map = snap_evt.blendshapes_as_dict() or {}
        lines.append(f"blendshapes={len(blendshape_map)}")
        if blendshape_map:
            top_blends = sorted(
                blendshape_map.items(), key=lambda kv: kv[1], reverse=True
            )[:8]
            for name, score in top_blends:
                lines.append(f"bs {name}={_fmt_num(score, 4)}")

        transform = snap_evt.transform_matrix_as_flat() or []
        lines.append(f"transform values={len(transform)}")
        if len(transform) >= 16:
            for r in range(4):
                row = transform[r * 4 : (r + 1) * 4]
                lines.append(
                    "m" + str(r) + ": " + " ".join(_fmt_num(v, 4) for v in row)
                )

        eyes = snap_evt.eyes_dict()
        left_center = eyes.get("leftIrisCenter")
        right_center = eyes.get("rightIrisCenter")
        if left_center is not None and right_center is not None:
            lines.append(
                "irisCtr L(x,y,z)="
                f"{_fmt_num(left_center[0], 4)},{_fmt_num(left_center[1], 4)},{_fmt_num(left_center[2], 4)} "
                "R(x,y,z)="
                f"{_fmt_num(right_center[0], 4)},{_fmt_num(right_center[1], 4)},{_fmt_num(right_center[2], 4)}"
            )
            lines.append(
                "gazeYawPitch L="
                f"{_fmt_num(eyes.get('leftEyeGazeYaw'), 2)}/{_fmt_num(eyes.get('leftEyeGazePitch'), 2)} "
                "R="
                f"{_fmt_num(eyes.get('rightEyeGazeYaw'), 2)}/{_fmt_num(eyes.get('rightEyeGazePitch'), 2)}"
            )
    elif isinstance(snap_evt, dict):
        lines.append(f"type={snap_evt.get('type', 'mesh')}")
        lines.append(f"face={bool(snap_evt.get('hasFace'))} landmarks={len(landmarks)}")
        blendshapes = snap_evt.get("blendshapes")
        if isinstance(blendshapes, dict):
            lines.append(f"blendshapes={len(blendshapes)}")
        transform = snap_evt.get("transformMatrix")
        if isinstance(transform, list):
            lines.append(f"transform values={len(transform)}")
        eyes = snap_evt.get("eyes")
        if isinstance(eyes, dict):
            if isinstance(eyes.get("leftIrisCenter"), list) and isinstance(
                eyes.get("rightIrisCenter"), list
            ):
                lc = eyes.get("leftIrisCenter")
                rc = eyes.get("rightIrisCenter")
                lines.append(
                    "irisCtr L(x,y,z)="
                    f"{_fmt_num(lc[0], 4)},{_fmt_num(lc[1], 4)},{_fmt_num(lc[2], 4)} "
                    "R(x,y,z)="
                    f"{_fmt_num(rc[0], 4)},{_fmt_num(rc[1], 4)},{_fmt_num(rc[2], 4)}"
                )
            l_gaze_yaw = eyes.get("leftEyeGazeYaw")
            l_gaze_pitch = eyes.get("leftEyeGazePitch")
            r_gaze_yaw = eyes.get("rightEyeGazeYaw")
            r_gaze_pitch = eyes.get("rightEyeGazePitch")
            if (
                l_gaze_yaw is not None
                or l_gaze_pitch is not None
                or r_gaze_yaw is not None
                or r_gaze_pitch is not None
            ):
                lines.append(
                    "gazeYawPitch L="
                    f"{_fmt_num(l_gaze_yaw, 2)}/{_fmt_num(l_gaze_pitch, 2)} "
                    "R="
                    f"{_fmt_num(r_gaze_yaw, 2)}/{_fmt_num(r_gaze_pitch, 2)}"
                )
    else:
        lines.append("event: none")
        lines.append(f"landmarks={len(landmarks)}")

    if len(landmarks) > 473:
        lxy = _safe_lm_xy(landmarks[468])
        rxy = _safe_lm_xy(landmarks[473])
        if lxy and rxy:
            lines.append(
                "iris L(x,y)="
                f"{_fmt_num(lxy[0], 4)},{_fmt_num(lxy[1], 4)} "
                "R(x,y)="
                f"{_fmt_num(rxy[0], 4)},{_fmt_num(rxy[1], 4)}"
            )

    return lines


def _draw_info_panel(img, lines: List[str]) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    pad = 8
    line_h = 18
    max_lines = max(1, (img.shape[0] - 24) // line_h)
    draw_lines = lines[:max_lines]
    if len(lines) > max_lines:
        draw_lines[-1] = f"... +{len(lines) - max_lines + 1} more"

    widths = [
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in draw_lines
    ]
    box_w = (
        min(max(widths) + pad * 2, img.shape[1] - 20)
        if widths
        else min(260, img.shape[1] - 20)
    )
    box_h = line_h * len(draw_lines) + pad * 2
    x0, y0 = 10, 10
    x1, y1 = x0 + box_w, y0 + box_h

    cv2.rectangle(img, (x0, y0), (x1, y1), HUD_BG, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), HUD_BORDER, 1, cv2.LINE_AA)

    for i, line in enumerate(draw_lines):
        tx = x0 + pad
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (tx + 1, ty + 1), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (tx, ty), font, scale, HUD_TEXT, thickness, cv2.LINE_AA)


def _draw_face_direction_from_ypr(
    img,
    origin: Tuple[int, int],
    yaw_deg: Any,
    pitch_deg: Any,
    roll_deg: Any,
) -> None:
    """Draw face forward vector from yaw/pitch/roll at given origin.

    Coordinate System Convention:
    - Yaw: Positive values indicate turning RIGHT, negative values indicate turning LEFT
    - Pitch: Positive values indicate tilting UP, negative values indicate tilting DOWN
    - Roll: Positive values indicate rotating counter-clockwise (tipping left)

    The displayed arrow shows the direction the face is looking, with:
    - Cyan arrow for face direction vector
    - Orange tick mark for roll indication
    """
    yaw = safe_float(yaw_deg, float("nan"))
    pitch = safe_float(pitch_deg, float("nan"))
    roll = safe_float(roll_deg, float("nan"))
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        return

    yaw_r = math.radians(yaw)
    pitch_r = math.radians(pitch)

    # Reconstruct normalized face-forward direction from yaw/pitch conventions.
    # Right-positive yaw: fx = sin(yaw) (positive x when looking right)
    # Up-positive pitch: fy = -sin(pitch) (negative y in image coords when looking up)
    fx = math.sin(yaw_r)
    fy = -math.sin(pitch_r)
    fz = -math.cos(yaw_r) * math.cos(pitch_r)
    mag = math.sqrt(fx * fx + fy * fy + fz * fz)
    if mag <= 1e-9:
        return
    fx /= mag
    fy /= mag

    h, w = img.shape[:2]
    length = int(max(36, min(w, h) * 0.16))
    ox, oy = int(origin[0]), int(origin[1])
    dx = fx * float(length)
    dy = fy * float(length)
    raw_len = math.hypot(dx, dy)
    min_vis = max(14.0, float(length) * 0.3)
    if raw_len > 1e-6 and raw_len < min_vis:
        s = min_vis / raw_len
        dx *= s
        dy *= s
    ex = int(round(ox + dx))
    ey = int(round(oy + dy))

    cv2.arrowedLine(img, (ox, oy), (ex, ey), WHITE, 4, cv2.LINE_AA, tipLength=0.22)
    cv2.arrowedLine(img, (ox, oy), (ex, ey), CYAN, 2, cv2.LINE_AA, tipLength=0.22)

    # Roll rotates around the forward axis; show it as a short nose-centered tick.
    if math.isfinite(roll):
        rr = math.radians(roll)
        tx = math.cos(rr)
        ty = -math.sin(rr)
        tick_len = max(10, int(length * 0.25))
        p1 = (
            int(round(ox - tx * tick_len * 0.5)),
            int(round(oy - ty * tick_len * 0.5)),
        )
        p2 = (
            int(round(ox + tx * tick_len * 0.5)),
            int(round(oy + ty * tick_len * 0.5)),
        )
        cv2.line(img, p1, p2, WHITE, 3, cv2.LINE_AA)
        cv2.line(img, p1, p2, ORANGE, 1, cv2.LINE_AA)


def render_camera_capture_marked(
    png_path: str,
    snap: Dict,
    overlay_w: float,
    overlay_h: float,
    click_pos: Tuple[float, float],
    draw_click: bool = True,
    draw_info_panel: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Render camera frame with face mesh landmarks and save to file."""
    img, err = build_camera_capture_marked_image(
        snap,
        overlay_w=overlay_w,
        overlay_h=overlay_h,
        click_pos=click_pos,
        draw_click=draw_click,
        draw_info_panel=draw_info_panel,
    )
    if img is None:
        return False, err or "No camera frame available yet."
    ok = cv2.imwrite(str(png_path), img)
    return (ok, None) if ok else (False, "cv2.imwrite failed")


def build_camera_capture_marked_image(
    snap: Dict,
    overlay_w: float,
    overlay_h: float,
    click_pos: Tuple[float, float],
    draw_click: bool = True,
    draw_info_panel: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Build marked camera frame with face mesh, ovals, and vectors."""
    frame = snap.get("frame")
    if frame is None:
        return None, "No camera frame available yet."

    mirror_view = True
    img = cv2.flip(frame, 1) if mirror_view else frame.copy()
    fh, fw = img.shape[:2]
    landmarks = snap.get("landmarks") or []
    snap_evt = snap.get("evt")

    # Draw mesh connections first, then points for clarity.
    if landmarks:
        for conn in FACE_MESH_CONNECTIONS:
            a = int(conn.start)
            b = int(conn.end)
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            p1 = _lm_to_px(landmarks[a], fw, fh, mirror_x=mirror_view)
            p2 = _lm_to_px(landmarks[b], fw, fh, mirror_x=mirror_view)
            cv2.line(img, p1, p2, (45, 140, 45), 1, cv2.LINE_AA)

        for lm in landmarks:
            cv2.circle(
                img,
                _lm_to_px(lm, fw, fh, mirror_x=mirror_view),
                1,
                GREEN,
                -1,
                cv2.LINE_AA,
            )

    # Draw explicit eye geometry from raw landmarks.
    left_iris_ring = _lm_points_px(
        landmarks, LEFT_IRIS_RING_IDXS, fw, fh, mirror_x=mirror_view
    )
    right_iris_ring = _lm_points_px(
        landmarks, RIGHT_IRIS_RING_IDXS, fw, fh, mirror_x=mirror_view
    )

    if 0 <= LEFT_IRIS_CENTER_IDX < len(landmarks):
        left_center = _lm_to_px(
            landmarks[LEFT_IRIS_CENTER_IDX], fw, fh, mirror_x=mirror_view
        )
    else:
        left_center = _center_px(left_iris_ring)
    if 0 <= RIGHT_IRIS_CENTER_IDX < len(landmarks):
        right_center = _lm_to_px(
            landmarks[RIGHT_IRIS_CENTER_IDX], fw, fh, mirror_x=mirror_view
        )
    else:
        right_center = _center_px(right_iris_ring)
    if left_center is not None:
        cv2.circle(img, left_center, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, left_center, 3, MAGENTA, -1, cv2.LINE_AA)
    if right_center is not None:
        cv2.circle(img, right_center, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, right_center, 3, MAGENTA, -1, cv2.LINE_AA)

    # Canthus points + eye axis used for lateral iris distance.
    def _draw_canthus_axis(inner_idx: int, outer_idx: int, iris_center, color):
        if inner_idx >= len(landmarks) or outer_idx >= len(landmarks):
            return
        inner_px = _lm_to_px(landmarks[inner_idx], fw, fh, mirror_x=mirror_view)
        outer_px = _lm_to_px(landmarks[outer_idx], fw, fh, mirror_x=mirror_view)
        cv2.line(img, inner_px, outer_px, WHITE, 3, cv2.LINE_AA)
        cv2.line(img, inner_px, outer_px, color, 1, cv2.LINE_AA)
        cv2.circle(img, inner_px, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, inner_px, 3, color, -1, cv2.LINE_AA)
        cv2.circle(img, outer_px, 4, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, outer_px, 2, color, -1, cv2.LINE_AA)

        if iris_center is None:
            return
        ax_dx = float(outer_px[0] - inner_px[0])
        ax_dy = float(outer_px[1] - inner_px[1])
        axis_len = math.hypot(ax_dx, ax_dy)
        if axis_len <= 1e-6:
            return
        ux = ax_dx / axis_len
        uy = ax_dy / axis_len
        ix = float(iris_center[0])
        iy = float(iris_center[1])
        t = (ix - inner_px[0]) * ux + (iy - inner_px[1]) * uy
        foot = (
            int(round(inner_px[0] + t * ux)),
            int(round(inner_px[1] + t * uy)),
        )
        # Iris -> inner canthus, parallel to inner->outer canthus axis
        # (drop iris perpendicularly onto the eye axis, then walk along
        # the axis back to the inner canthus -- the lateral distance segment).
        iris_px = (int(round(ix)), int(round(iy)))
        cv2.line(img, iris_px, foot, WHITE, 2, cv2.LINE_AA)
        cv2.line(img, iris_px, foot, color, 1, cv2.LINE_AA)
        cv2.line(img, foot, inner_px, WHITE, 3, cv2.LINE_AA)
        cv2.line(img, foot, inner_px, color, 1, cv2.LINE_AA)
        cv2.circle(img, foot, 3, WHITE, -1, cv2.LINE_AA)

    _draw_canthus_axis(LEFT_EYE_INNER_IDX, LEFT_EYE_OUTER_IDX, left_center, ORANGE)
    _draw_canthus_axis(RIGHT_EYE_INNER_IDX, RIGHT_EYE_OUTER_IDX, right_center, CYAN)

    # Nose T: bridge->base axis (vertical reference for eye pitch) and
    # the perpendicular at the bridge (horizontal eye-span direction).
    nose_bridge_px = (
        _lm_to_px(landmarks[NOSE_BRIDGE_IDX], fw, fh, mirror_x=mirror_view)
        if NOSE_BRIDGE_IDX < len(landmarks)
        else None
    )
    nose_base_px = (
        _lm_to_px(landmarks[NOSE_BASE_IDX], fw, fh, mirror_x=mirror_view)
        if NOSE_BASE_IDX < len(landmarks)
        else None
    )
    if nose_bridge_px is not None and nose_base_px is not None:
        bx, by = float(nose_bridge_px[0]), float(nose_bridge_px[1])
        nbx, nby = float(nose_base_px[0]), float(nose_base_px[1])
        ax = nbx - bx
        ay = nby - by
        axis_len = math.hypot(ax, ay)
        if axis_len > 1e-6:
            n_hat_x = ax / axis_len
            n_hat_y = ay / axis_len
            p_hat_x = -n_hat_y
            p_hat_y = n_hat_x
            if left_center is not None and right_center is not None:
                eye_span = math.hypot(
                    float(right_center[0] - left_center[0]),
                    float(right_center[1] - left_center[1]),
                )
            else:
                eye_span = axis_len
            half_len = max(90.0, eye_span * 2.5)
            p1 = (
                int(round(bx - p_hat_x * half_len)),
                int(round(by - p_hat_y * half_len)),
            )
            p2 = (
                int(round(bx + p_hat_x * half_len)),
                int(round(by + p_hat_y * half_len)),
            )
            cv2.line(img, nose_bridge_px, nose_base_px, WHITE, 4, cv2.LINE_AA)
            cv2.line(img, nose_bridge_px, nose_base_px, RED, 2, cv2.LINE_AA)
            cv2.line(img, p1, p2, WHITE, 4, cv2.LINE_AA)
            cv2.line(img, p1, p2, GREEN, 2, cv2.LINE_AA)

            # Iris -> bridge-perpendicular line, perpendicular drop
            # (vertical-distance segment from iris to the nose T crossbar).
            def _drop_to_bridge_perp(iris_center, color):
                if iris_center is None:
                    return
                ix = float(iris_center[0])
                iy = float(iris_center[1])
                proj_t = (ix - bx) * p_hat_x + (iy - by) * p_hat_y
                foot = (
                    int(round(bx + proj_t * p_hat_x)),
                    int(round(by + proj_t * p_hat_y)),
                )
                iris_px = (int(round(ix)), int(round(iy)))
                cv2.line(img, iris_px, foot, WHITE, 2, cv2.LINE_AA)
                cv2.line(img, iris_px, foot, color, 1, cv2.LINE_AA)
                cv2.circle(img, foot, 3, WHITE, -1, cv2.LINE_AA)

            _drop_to_bridge_perp(left_center, ORANGE)
            _drop_to_bridge_perp(right_center, CYAN)
        cv2.circle(img, nose_bridge_px, 6, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, nose_bridge_px, 4, RED, -1, cv2.LINE_AA)
        cv2.circle(img, nose_base_px, 6, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, nose_base_px, 4, YELLOW, -1, cv2.LINE_AA)

    # Click marker mapped from overlay-space to frame-space.
    cx = int(
        clamp(round((float(click_pos[0]) / max(1.0, float(overlay_w))) * fw), 0, fw - 1)
    )
    cy = int(
        clamp(round((float(click_pos[1]) / max(1.0, float(overlay_h))) * fh), 0, fh - 1)
    )
    if draw_click:
        cv2.circle(img, (cx, cy), 14, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, RED, -1, cv2.LINE_AA)

    if draw_info_panel:
        _draw_info_panel(img, _build_event_lines(snap_evt, landmarks))
    return img, None


def save_capture(
    display: Dict,
    w: float,
    h: float,
    click_pos: Tuple[float, float],
    frame: Any,
    evt: Any,
    runtime_evt: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a capture payload that supports offline calibration diagnostics."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    ts = ms_now()
    click_x = clamp(float(click_pos[0]), 0.0, w - 1.0)
    click_y = clamp(float(click_pos[1]), 0.0, h - 1.0)

    base = f"mesh_capture_{ts}"
    png_path = CAPTURE_DIR / f"{base}.png"
    raw_png_path = CAPTURE_DIR / f"{base}_raw.png"
    eye_debug_dir = CAPTURE_DIR / f"{base}_eye_debug"
    json_path = CAPTURE_DIR / f"{base}.json"

    snap_evt = evt
    snap_frame = frame
    snap_landmarks = list(evt.landmarks) if evt and evt.landmarks else None
    snap = {"evt": snap_evt, "frame": snap_frame, "landmarks": snap_landmarks}

    if isinstance(snap_evt, FaceMeshEvent):
        event_dump = snap_evt.to_capture_dump()
        mesh_data = event_dump.get("meshData") or {}
    elif isinstance(snap_evt, dict):
        event_dump = snap_evt
        eyes = snap_evt.get("eyes")
        mesh_data = {
            "landmarks": snap_evt.get("landmarks"),
            "blendshapes": snap_evt.get("blendshapes"),
            "transformMatrix": snap_evt.get("transformMatrix"),
            "faceMaskSegment": snap_evt.get("faceMaskSegment"),
            "eyes": eyes,
        }
    else:
        event_dump = None
        mesh_data = {
            "landmarks": None,
            "blendshapes": None,
            "transformMatrix": None,
            "faceMaskSegment": None,
            "eyes": None,
        }

    shot_ok, shot_err = render_camera_capture_marked(
        str(png_path), snap, overlay_w=w, overlay_h=h, click_pos=(click_x, click_y)
    )
    raw_ok = False
    raw_err: Optional[str] = None
    if snap_frame is not None:
        mirrored_raw = cv2.flip(snap_frame, 1)
        raw_ok = bool(cv2.imwrite(str(raw_png_path), mirrored_raw))
        if not raw_ok:
            raw_err = "cv2.imwrite failed"
    else:
        raw_err = "No camera frame available yet."

    payload = {
        "timestamp": ts,
        "click": {
            "xWindow": click_x,
            "yWindow": click_y,
            "xScreen": float(display["x"]) + click_x,
            "yScreen": float(display["y"]) + click_y,
        },
        "faceMeshEvent": event_dump,
        "runtimeEvent": runtime_evt,
        "calibratedGaze": (
            {
                "faceDeltaYaw": runtime_evt.get("face_delta_yaw"),
                "faceDeltaPitch": runtime_evt.get("face_delta_pitch"),
                "correctedEyeYaw": runtime_evt.get("corrected_eye_yaw"),
                "correctedEyePitch": runtime_evt.get("corrected_eye_pitch"),
                "correctedYaw": runtime_evt.get("corrected_yaw"),
                "correctedPitch": runtime_evt.get("corrected_pitch"),
                "correctedYawLinear": runtime_evt.get("corrected_yaw_linear"),
                "correctedPitchLinear": runtime_evt.get("corrected_pitch_linear"),
                "correctedScreenX": runtime_evt.get("corrected_screen_x"),
                "correctedScreenY": runtime_evt.get("corrected_screen_y"),
                "overlayX": runtime_evt.get("overlay_x"),
                "overlayY": runtime_evt.get("overlay_y"),
            }
            if isinstance(runtime_evt, dict)
            else None
        ),
        "calibrationModel": (
            {
                "centerEyeYaw": runtime_evt.get("center_eye_yaw"),
                "centerEyePitch": runtime_evt.get("center_eye_pitch"),
                "faceCenterYaw": runtime_evt.get("face_center_yaw"),
                "faceCenterPitch": runtime_evt.get("face_center_pitch"),
                "faceCenterX": runtime_evt.get("face_center_x"),
                "faceCenterY": runtime_evt.get("face_center_y"),
                "faceCenterZ": runtime_evt.get("face_center_z"),
                "centerZeta": runtime_evt.get("center_zeta"),
                "yawCoefficientPositive": runtime_evt.get("yaw_coefficient_positive"),
                "yawCoefficientNegative": runtime_evt.get("yaw_coefficient_negative"),
                "pitchCoefficientPositive": runtime_evt.get(
                    "pitch_coefficient_positive"
                ),
                "pitchCoefficientNegative": runtime_evt.get(
                    "pitch_coefficient_negative"
                ),
                "yawFromPitchCoupling": runtime_evt.get("yaw_from_pitch_coupling"),
                "pitchFromYawCoupling": runtime_evt.get("pitch_from_yaw_coupling"),
                "eyeYawMin": runtime_evt.get("eye_yaw_min"),
                "eyeYawMax": runtime_evt.get("eye_yaw_max"),
                "eyePitchMin": runtime_evt.get("eye_pitch_min"),
                "eyePitchMax": runtime_evt.get("eye_pitch_max"),
                "screenCenterCamX": runtime_evt.get("screen_center_cam_x"),
                "screenCenterCamY": runtime_evt.get("screen_center_cam_y"),
                "screenCenterCamZ": runtime_evt.get("screen_center_cam_z"),
                "screenAxisXX": runtime_evt.get("screen_axis_x_x"),
                "screenAxisXY": runtime_evt.get("screen_axis_x_y"),
                "screenAxisXZ": runtime_evt.get("screen_axis_x_z"),
                "screenAxisYX": runtime_evt.get("screen_axis_y_x"),
                "screenAxisYY": runtime_evt.get("screen_axis_y_y"),
                "screenAxisYZ": runtime_evt.get("screen_axis_y_z"),
                "screenScaleX": runtime_evt.get("screen_scale_x"),
                "screenScaleY": runtime_evt.get("screen_scale_y"),
                "screenFitRmse": runtime_evt.get("screen_fit_rmse"),
            }
            if isinstance(runtime_evt, dict)
            else None
        ),
        "displayGeometry": (
            {
                "originX": runtime_evt.get("origin_x"),
                "originY": runtime_evt.get("origin_y"),
                "width": runtime_evt.get("display_width"),
                "height": runtime_evt.get("display_height"),
            }
            if isinstance(runtime_evt, dict)
            else None
        ),
        "meshData": mesh_data,
        "screenshot": {
            "path": str(png_path),
            "ok": shot_ok,
            "error": shot_err,
        },
        "rawCameraScreenshot": {
            "path": str(raw_png_path),
            "ok": raw_ok,
            "error": raw_err,
        },
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if shot_ok and raw_ok:
        print(
            f"Saved {png_path.name}, {raw_png_path.name}, and {json_path.name}",
            flush=True,
        )
    elif shot_ok and not raw_ok:
        print(
            f"Saved {png_path.name} and {json_path.name} (raw frame failed: {raw_err})",
            flush=True,
        )
    else:
        print(
            f"Marked screenshot failed, saved JSON: {json_path.name} "
            f"(marked={shot_err}, raw={'ok' if raw_ok else raw_err})",
            flush=True,
        )
