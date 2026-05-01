"""
Shared gaze primitive projection and rendering utilities.
"""

from dataclasses import dataclass
import functools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .facemesh_dao import clamp, safe_float


COMPONENT_FACE = "face"
COMPONENT_EYE = "eye"
COMPONENT_SUM = "sum"
SOURCE_RAW = "raw"
SOURCE_CORRELATED = "correlated"

RGB_FACE = (255, 40, 40)
RGB_EYE = (70, 180, 255)
RGB_SUM = (80, 230, 120)

_RGB_BY_COMPONENT = {
    COMPONENT_FACE: RGB_FACE,
    COMPONENT_EYE: RGB_EYE,
    COMPONENT_SUM: RGB_SUM,
}


@dataclass(frozen=True)
class GazePrimitive:
    """Describe one projected gaze primitive with source and component semantics."""

    source: str
    component: str
    point: Tuple[int, int]


def _positive_or(value: Any, fallback: float) -> float:
    v = safe_float(value, fallback)
    return float(v) if float(v) > 1e-9 else float(fallback)


def _finite_float(value: Any) -> Optional[float]:
    v = safe_float(value, float("nan"))
    if v != v:
        return None
    return float(v)


def _origin(
    width: int,
    height: int,
    origin_x: Optional[float],
    origin_y: Optional[float],
) -> Tuple[float, float]:
    ox = float(width) * 0.5 if origin_x is None else float(origin_x)
    oy = float(height) * 0.5 if origin_y is None else float(origin_y)
    return ox, oy


def _angles_to_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cp = math.cos(pitch)
    return np.array(
        [
            math.sin(yaw) * cp,
            -math.sin(pitch),
            -math.cos(yaw) * cp,
        ],
        dtype=float,
    )


def _project_to_screen_offsets(
    head_origin: np.ndarray,
    direction: np.ndarray,
    screen_center: np.ndarray,
    screen_axis_x: np.ndarray,
    screen_axis_y: np.ndarray,
) -> Optional[Tuple[float, float]]:
    # screen_axis_x / screen_axis_y are guaranteed orthonormal by
    # `_screen_geometry_from_values` (gram-schmidt + normalize), so the basis
    # inverse is just the transpose: offsets = [axis_x . d, axis_y . d].
    normal = np.cross(screen_axis_x, screen_axis_y)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return None
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= 1e-9:
        return None
    ray_t = float(np.dot(normal, (screen_center - head_origin)) / denominator)
    if ray_t <= 1e-9:
        return None
    intersection = head_origin + ray_t * direction
    delta = intersection - screen_center
    return float(np.dot(screen_axis_x, delta)), float(np.dot(screen_axis_y, delta))


def _project_to_screen_offsets_with_t(
    head_origin: np.ndarray,
    direction: np.ndarray,
    screen_center: np.ndarray,
    screen_axis_x: np.ndarray,
    screen_axis_y: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    # See `_project_to_screen_offsets`: orthonormal basis lets us skip lstsq.
    normal = np.cross(screen_axis_x, screen_axis_y)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return None
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= 1e-9:
        return None
    ray_t = float(np.dot(normal, (screen_center - head_origin)) / denominator)
    if ray_t <= 1e-9:
        return None
    intersection = head_origin + ray_t * direction
    delta = intersection - screen_center
    return (
        float(np.dot(screen_axis_x, delta)),
        float(np.dot(screen_axis_y, delta)),
        ray_t,
    )


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return fallback
    return v / norm


def _screen_geometry_from_evt(
    evt: Dict[str, Any],
    *,
    head_x: float,
    head_y: float,
    head_z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _screen_geometry_from_values(
        head_x=head_x,
        head_y=head_y,
        head_z=head_z,
        center_zeta=evt.get("center_zeta"),
        screen_center_cam_x=evt.get("screen_center_cam_x"),
        screen_center_cam_y=evt.get("screen_center_cam_y"),
        screen_center_cam_z=evt.get("screen_center_cam_z"),
        screen_axis_x_x=evt.get("screen_axis_x_x"),
        screen_axis_x_y=evt.get("screen_axis_x_y"),
        screen_axis_x_z=evt.get("screen_axis_x_z"),
        screen_axis_y_x=evt.get("screen_axis_y_x"),
        screen_axis_y_y=evt.get("screen_axis_y_y"),
        screen_axis_y_z=evt.get("screen_axis_y_z"),
        screen_fit_rmse=evt.get("screen_fit_rmse"),
    )


@functools.lru_cache(maxsize=8)
def _calibrated_screen_geometry(
    screen_center_cam_x: float,
    screen_center_cam_y: float,
    screen_center_cam_z: float,
    screen_axis_x_x: float,
    screen_axis_x_y: float,
    screen_axis_x_z: float,
    screen_axis_y_x: float,
    screen_axis_y_y: float,
    screen_axis_y_z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Calibration constants don't change frame-to-frame; cache the derived
    # axes/normal so the per-frame call site is a dict lookup.
    screen_center = np.array(
        [screen_center_cam_x, screen_center_cam_y, screen_center_cam_z],
        dtype=float,
    )
    raw_x = np.array(
        [screen_axis_x_x, screen_axis_x_y, screen_axis_x_z], dtype=float
    )
    raw_y = np.array(
        [screen_axis_y_x, screen_axis_y_y, screen_axis_y_z], dtype=float
    )
    axis_x = _normalize(raw_x, np.array([1.0, 0.0, 0.0], dtype=float))
    axis_y_ortho = raw_y - float(np.dot(raw_y, axis_x)) * axis_x
    axis_y = _normalize(axis_y_ortho, np.array([0.0, 1.0, 0.0], dtype=float))
    raw_normal = np.cross(axis_x, axis_y)
    normal = _normalize(raw_normal, np.array([0.0, 0.0, -1.0], dtype=float))
    # Returned arrays are shared with future callers via the cache; freeze
    # them so an accidental mutation can't poison every frame after.
    for arr in (screen_center, axis_x, axis_y, normal):
        arr.setflags(write=False)
    return screen_center, axis_x, axis_y, normal


def _screen_geometry_from_values(
    *,
    head_x: Any,
    head_y: Any,
    head_z: Any,
    center_zeta: Any,
    screen_center_cam_x: Any,
    screen_center_cam_y: Any,
    screen_center_cam_z: Any,
    screen_axis_x_x: Any,
    screen_axis_x_y: Any,
    screen_axis_x_z: Any,
    screen_axis_y_x: Any,
    screen_axis_y_y: Any,
    screen_axis_y_z: Any,
    screen_fit_rmse: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_rmse = _finite_float(screen_fit_rmse)
    center_zeta_f = _positive_or(center_zeta, 1200.0)
    if fit_rmse is None or fit_rmse < 0.0:
        head_x_f = safe_float(head_x, 0.0)
        head_y_f = safe_float(head_y, 0.0)
        head_z_f = _positive_or(head_z, center_zeta_f)
        depth = _positive_or(head_z_f, center_zeta_f)
        screen_center = np.array([head_x_f, head_y_f, head_z_f - depth], dtype=float)
        return (
            screen_center,
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
        )

    screen_center, axis_x, axis_y, _ = _calibrated_screen_geometry(
        safe_float(screen_center_cam_x, 0.0),
        safe_float(screen_center_cam_y, 0.0),
        safe_float(screen_center_cam_z, center_zeta_f),
        safe_float(screen_axis_x_x, 1.0),
        safe_float(screen_axis_x_y, 0.0),
        safe_float(screen_axis_x_z, 0.0),
        safe_float(screen_axis_y_x, 0.0),
        safe_float(screen_axis_y_y, 1.0),
        safe_float(screen_axis_y_z, 0.0),
    )
    return screen_center, axis_x, axis_y


def project_head_angles_to_screen_xy(
    *,
    yaw_deg: Any,
    pitch_deg: Any,
    head_x: Any,
    head_y: Any,
    head_z: Any,
    center_zeta: Any,
    screen_center_cam_x: Any,
    screen_center_cam_y: Any,
    screen_center_cam_z: Any,
    screen_axis_x_x: Any,
    screen_axis_x_y: Any,
    screen_axis_x_z: Any,
    screen_axis_y_x: Any,
    screen_axis_y_y: Any,
    screen_axis_y_z: Any,
    screen_fit_rmse: Any = -1.0,
    screen_scale_x: Any = 1.0,
    screen_scale_y: Any = 1.0,
    origin_x: Any = None,
    origin_y: Any = None,
) -> Optional[Dict[str, float]]:
    yaw = _finite_float(yaw_deg)
    pitch = _finite_float(pitch_deg)
    head_x_f = _finite_float(head_x)
    head_y_f = _finite_float(head_y)
    head_z_f = _finite_float(head_z)
    if yaw is None or pitch is None:
        return None
    if head_x_f is None or head_y_f is None or head_z_f is None:
        return None

    screen_center, screen_axis_x, screen_axis_y = _screen_geometry_from_values(
        head_x=head_x_f,
        head_y=head_y_f,
        head_z=head_z_f,
        center_zeta=center_zeta,
        screen_center_cam_x=screen_center_cam_x,
        screen_center_cam_y=screen_center_cam_y,
        screen_center_cam_z=screen_center_cam_z,
        screen_axis_x_x=screen_axis_x_x,
        screen_axis_x_y=screen_axis_x_y,
        screen_axis_x_z=screen_axis_x_z,
        screen_axis_y_x=screen_axis_y_x,
        screen_axis_y_y=screen_axis_y_y,
        screen_axis_y_z=screen_axis_y_z,
        screen_fit_rmse=screen_fit_rmse,
    )

    projected = _project_to_screen_offsets_with_t(
        head_origin=np.array([head_x_f, head_y_f, head_z_f], dtype=float),
        direction=_angles_to_direction(yaw, pitch),
        screen_center=screen_center,
        screen_axis_x=screen_axis_x,
        screen_axis_y=screen_axis_y,
    )
    if projected is None:
        return None
    offset_x, offset_y, projection_t = projected
    scale_x = _positive_or(screen_scale_x, 1.0)
    scale_y = _positive_or(screen_scale_y, 1.0)
    origin_x_f = safe_float(origin_x, 0.0)
    origin_y_f = safe_float(origin_y, 0.0)
    offset_x_px = float(offset_x * scale_x)
    offset_y_px = float(offset_y * scale_y)
    return {
        "screen_x": float(origin_x_f + offset_x_px),
        "screen_y": float(origin_y_f + offset_y_px),
        "offset_x": float(offset_x_px),
        "offset_y": float(offset_y_px),
        "projection_t": float(projection_t),
    }


def screen_xy_to_head_angles(
    *,
    screen_x: Any,
    screen_y: Any,
    head_x: Any,
    head_y: Any,
    head_z: Any,
    center_zeta: Any,
    screen_center_cam_x: Any,
    screen_center_cam_y: Any,
    screen_center_cam_z: Any,
    screen_axis_x_x: Any,
    screen_axis_x_y: Any,
    screen_axis_x_z: Any,
    screen_axis_y_x: Any,
    screen_axis_y_y: Any,
    screen_axis_y_z: Any,
    screen_fit_rmse: Any = -1.0,
    screen_scale_x: Any = 1.0,
    screen_scale_y: Any = 1.0,
    origin_x: Any = None,
    origin_y: Any = None,
) -> Optional[Tuple[float, float]]:
    sx = _finite_float(screen_x)
    sy = _finite_float(screen_y)
    head_x_f = _finite_float(head_x)
    head_y_f = _finite_float(head_y)
    head_z_f = _finite_float(head_z)
    if sx is None or sy is None:
        return None
    if head_x_f is None or head_y_f is None or head_z_f is None:
        return None

    screen_center, screen_axis_x, screen_axis_y = _screen_geometry_from_values(
        head_x=head_x_f,
        head_y=head_y_f,
        head_z=head_z_f,
        center_zeta=center_zeta,
        screen_center_cam_x=screen_center_cam_x,
        screen_center_cam_y=screen_center_cam_y,
        screen_center_cam_z=screen_center_cam_z,
        screen_axis_x_x=screen_axis_x_x,
        screen_axis_x_y=screen_axis_x_y,
        screen_axis_x_z=screen_axis_x_z,
        screen_axis_y_x=screen_axis_y_x,
        screen_axis_y_y=screen_axis_y_y,
        screen_axis_y_z=screen_axis_y_z,
        screen_fit_rmse=screen_fit_rmse,
    )

    origin_x_f = safe_float(origin_x, 0.0)
    origin_y_f = safe_float(origin_y, 0.0)
    scale_x = _positive_or(screen_scale_x, 1.0)
    scale_y = _positive_or(screen_scale_y, 1.0)
    target = (
        screen_center
        + ((sx - origin_x_f) / scale_x) * screen_axis_x
        + ((sy - origin_y_f) / scale_y) * screen_axis_y
    )
    direction = target - np.array([head_x_f, head_y_f, head_z_f], dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        return None
    direction = direction / direction_norm
    sin_pitch = -float(direction[1])
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.degrees(math.asin(sin_pitch))
    yaw = math.degrees(math.atan2(float(direction[0]), -float(direction[2])))
    return float(yaw), float(pitch)


def project_xy_to_screen(
    x: Any,
    y: Any,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Clamp an input screen coordinate to target bounds."""
    sx = _finite_float(x)
    sy = _finite_float(y)
    if sx is None or sy is None:
        return None
    cx = clamp(sx, 0.0, float(max(0, width - 1)))
    cy = clamp(sy, 0.0, float(max(0, height - 1)))
    return int(round(cx)), int(round(cy))


def _project_from_head_and_angles(
    evt: Dict[str, Any],
    *,
    yaw_deg: Any,
    pitch_deg: Any,
    width: int,
    height: int,
    origin_x: Optional[float],
    origin_y: Optional[float],
) -> Optional[Tuple[int, int]]:
    yaw = _finite_float(yaw_deg)
    pitch = _finite_float(pitch_deg)
    if yaw is None or pitch is None:
        return None
    head_x = _finite_float(evt.get("head_x"))
    head_y = _finite_float(evt.get("head_y"))
    head_z = _finite_float(evt.get("head_z"))
    if head_x is None or head_y is None or head_z is None:
        return None
    event_origin_x = _finite_float(evt.get("origin_x"))
    event_origin_y = _finite_float(evt.get("origin_y"))
    ox, oy = _origin(
        width,
        height,
        event_origin_x if origin_x is None else origin_x,
        event_origin_y if origin_y is None else origin_y,
    )
    screen_center, screen_axis_x, screen_axis_y = _screen_geometry_from_evt(
        evt,
        head_x=head_x,
        head_y=head_y,
        head_z=head_z,
    )
    offsets = _project_to_screen_offsets(
        head_origin=np.array([head_x, head_y, head_z], dtype=float),
        direction=_angles_to_direction(yaw, pitch),
        screen_center=screen_center,
        screen_axis_x=screen_axis_x,
        screen_axis_y=screen_axis_y,
    )
    if offsets is None:
        return None
    offset_x, offset_y = offsets
    scale_x = _positive_or(evt.get("screen_scale_x"), 1.0)
    scale_y = _positive_or(evt.get("screen_scale_y"), 1.0)
    return project_xy_to_screen(
        ox + offset_x * scale_x,
        oy + offset_y * scale_y,
        width,
        height,
    )


def _raw_sum_angles(evt: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    head_yaw = _finite_float(evt.get("head_yaw"))
    head_pitch = _finite_float(evt.get("head_pitch"))
    eye_yaw = _finite_float(evt.get("raw_combined_eye_gaze_yaw"))
    eye_pitch = _finite_float(evt.get("raw_combined_eye_gaze_pitch"))
    if (
        head_yaw is None
        or head_pitch is None
        or eye_yaw is None
        or eye_pitch is None
    ):
        return None, None
    return head_yaw + eye_yaw, head_pitch + eye_pitch


def collect_gaze_primitives(
    evt: Optional[Dict[str, Any]],
    width: int,
    height: int,
    *,
    origin_x: Optional[float] = None,
    origin_y: Optional[float] = None,
) -> List[GazePrimitive]:
    """Create raw and correlated gaze primitives from runtime event payload."""
    if not evt:
        return []

    primitives: List[GazePrimitive] = []

    raw_face = _project_from_head_and_angles(
        evt,
        yaw_deg=evt.get("head_yaw"),
        pitch_deg=evt.get("head_pitch"),
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if raw_face is not None:
        primitives.append(
            GazePrimitive(source=SOURCE_RAW, component=COMPONENT_FACE, point=raw_face)
        )

    raw_eye = _project_from_head_and_angles(
        evt,
        yaw_deg=evt.get("raw_combined_eye_gaze_yaw"),
        pitch_deg=evt.get("raw_combined_eye_gaze_pitch"),
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if raw_eye is not None:
        primitives.append(
            GazePrimitive(source=SOURCE_RAW, component=COMPONENT_EYE, point=raw_eye)
        )

    raw_sum_yaw, raw_sum_pitch = _raw_sum_angles(evt)
    raw_sum = _project_from_head_and_angles(
        evt,
        yaw_deg=raw_sum_yaw,
        pitch_deg=raw_sum_pitch,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if raw_sum is not None:
        primitives.append(
            GazePrimitive(source=SOURCE_RAW, component=COMPONENT_SUM, point=raw_sum)
        )

    correlated_face = _project_from_head_and_angles(
        evt,
        yaw_deg=evt.get("face_delta_yaw"),
        pitch_deg=evt.get("face_delta_pitch"),
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if correlated_face is not None:
        primitives.append(
            GazePrimitive(
                source=SOURCE_CORRELATED,
                component=COMPONENT_FACE,
                point=correlated_face,
            )
        )

    correlated_eye = _project_from_head_and_angles(
        evt,
        yaw_deg=evt.get("corrected_eye_yaw"),
        pitch_deg=evt.get("corrected_eye_pitch"),
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if correlated_eye is not None:
        primitives.append(
            GazePrimitive(
                source=SOURCE_CORRELATED,
                component=COMPONENT_EYE,
                point=correlated_eye,
            )
        )

    correlated_sum = _project_from_head_and_angles(
        evt,
        yaw_deg=evt.get("corrected_yaw_linear"),
        pitch_deg=evt.get("corrected_pitch_linear"),
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if correlated_sum is None:
        correlated_sum = _project_from_head_and_angles(
            evt,
            yaw_deg=evt.get("corrected_yaw"),
            pitch_deg=evt.get("corrected_pitch"),
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
        )
    if correlated_sum is None:
        correlated_sum = project_xy_to_screen(
            evt.get("corrected_screen_x"),
            evt.get("corrected_screen_y"),
            width,
            height,
        )
    if correlated_sum is not None:
        primitives.append(
            GazePrimitive(
                source=SOURCE_CORRELATED,
                component=COMPONENT_SUM,
                point=correlated_sum,
            )
        )

    return primitives


def _ordered(primitives: Sequence[GazePrimitive]) -> List[GazePrimitive]:
    source_order = {SOURCE_CORRELATED: 0, SOURCE_RAW: 1}
    component_order = {COMPONENT_FACE: 0, COMPONENT_EYE: 1, COMPONENT_SUM: 2}
    return sorted(
        primitives,
        key=lambda p: (
            source_order.get(p.source, 99),
            component_order.get(p.component, 99),
        ),
    )


def draw_gaze_primitives_pygame(
    surface,
    primitives: Sequence[GazePrimitive],
    *,
    radius: int,
    outline_thickness: int = 2,
) -> None:
    """Render gaze primitives into a pygame surface."""
    import pygame

    for primitive in _ordered(primitives):
        color = _RGB_BY_COMPONENT.get(primitive.component)
        if color is None:
            continue
        px, py = primitive.point
        if primitive.source == SOURCE_RAW:
            pygame.draw.circle(
                surface, color, (int(px), int(py)), int(radius), int(outline_thickness)
            )
        else:
            pygame.draw.circle(surface, color, (int(px), int(py)), int(radius))


def draw_gaze_primitives_cv2(
    image,
    primitives: Sequence[GazePrimitive],
    *,
    radius: int,
    outline_thickness: int = 2,
) -> None:
    """Render gaze primitives into a cv2 image."""
    for primitive in _ordered(primitives):
        color_rgb = _RGB_BY_COMPONENT.get(primitive.component)
        if color_rgb is None:
            continue
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        px, py = primitive.point
        thickness = int(outline_thickness) if primitive.source == SOURCE_RAW else -1
        cv2.circle(
            image,
            (int(px), int(py)),
            int(radius),
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )
