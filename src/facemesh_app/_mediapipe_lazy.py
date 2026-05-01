"""Monkey-patch mediapipe's FaceLandmarkerResult.from_ctypes for lazy landmarks.

Mediapipe eagerly builds 478 NormalizedLandmark dataclass instances per frame
inside detect_for_video, even though our DAO touches ~12 specific indices.
This module replaces that with a single C-side memcpy into a numpy (n, 3) array
plus a list-like wrapper that materializes a NormalizedLandmark on indexing.

Why: ~1 ms/frame of Python work disappears when the patch is active.

Risk: tied to mediapipe's private C struct layout. We sanity-check the layout
at patch time and refuse to install if the structure has shifted; callers can
run the unpatched implementation as a fallback.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_lazy_landmarks_patch() -> bool:
    """Install the lazy-landmark patch. Idempotent. Returns True if patched."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        from mediapipe.tasks.python.components.containers import (
            category as category_lib,
        )
        from mediapipe.tasks.python.components.containers import (
            landmark as landmark_lib,
        )
        from mediapipe.tasks.python.components.containers import (
            landmark_c as landmark_c_lib,
        )
        from mediapipe.tasks.python.vision import face_landmarker
    except ImportError as exc:
        logger.warning("Lazy-landmark patch skipped: mediapipe import failed: %s", exc)
        return False

    NLC = landmark_c_lib.NormalizedLandmarkC

    # Layout sanity: the first three fields must be (x, y, z) as c_float at
    # offsets 0/4/8 — otherwise the bulk-copy below silently reads garbage.
    expected_head = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]
    if list(NLC._fields_)[:3] != expected_head:
        logger.warning(
            "Lazy-landmark patch skipped: NormalizedLandmarkC head is %s, expected %s",
            list(NLC._fields_)[:3],
            expected_head,
        )
        return False

    struct_size = ctypes.sizeof(NLC)
    xyz_dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [np.float32, np.float32, np.float32],
            "offsets": [0, 4, 8],
            "itemsize": struct_size,
        }
    )

    expected_result_fields = {
        "face_landmarks",
        "face_landmarks_count",
        "face_blendshapes",
        "face_blendshapes_count",
        "facial_transformation_matrixes",
        "facial_transformation_matrixes_count",
    }
    actual_result_fields = {name for name, _ in face_landmarker.FaceLandmarkerResultC._fields_}
    if not expected_result_fields.issubset(actual_result_fields):
        logger.warning(
            "Lazy-landmark patch skipped: FaceLandmarkerResultC layout drifted (have %s)",
            actual_result_fields,
        )
        return False

    NormalizedLandmark = landmark_lib.NormalizedLandmark
    Category = category_lib.Category

    class _LazyNormalizedLandmarkList:
        """Numpy-backed list-of-NormalizedLandmark with on-demand materialization."""

        __slots__ = ("_xyz",)

        def __init__(self, xyz: np.ndarray):
            self._xyz = xyz

        def __len__(self) -> int:
            return self._xyz.shape[0]

        def __getitem__(self, idx) -> NormalizedLandmark:
            row = self._xyz[idx]
            return NormalizedLandmark(x=float(row[0]), y=float(row[1]), z=float(row[2]))

        def __iter__(self) -> Iterator[NormalizedLandmark]:
            xyz = self._xyz
            for i in range(xyz.shape[0]):
                row = xyz[i]
                yield NormalizedLandmark(
                    x=float(row[0]), y=float(row[1]), z=float(row[2])
                )

        def __bool__(self) -> bool:
            return self._xyz.shape[0] > 0

    def _bulk_extract_xyz(landmarks_c) -> np.ndarray:
        n = int(landmarks_c.landmarks_count)
        xyz = np.empty((n, 3), dtype=np.float32)
        if n == 0:
            return xyz
        # ctypes.string_at performs one C-level memcpy. The c_char_p `name`
        # field is just bytes here — we never decode it.
        raw = ctypes.string_at(landmarks_c.landmarks, n * struct_size)
        view = np.frombuffer(raw, dtype=xyz_dtype, count=n)
        xyz[:, 0] = view["x"]
        xyz[:, 1] = view["y"]
        xyz[:, 2] = view["z"]
        return xyz

    def _patched_from_ctypes(cls, c_struct):
        face_landmarks = []
        for i in range(c_struct.face_landmarks_count):
            landmarks_c = c_struct.face_landmarks[i]
            face_landmarks.append(
                _LazyNormalizedLandmarkList(_bulk_extract_xyz(landmarks_c))
            )

        face_blendshapes = []
        for i in range(c_struct.face_blendshapes_count):
            categories_c = c_struct.face_blendshapes[i]
            face_blendshapes.append(
                [
                    Category.from_ctypes(categories_c.categories[j])
                    for j in range(categories_c.categories_count)
                ]
            )

        facial_transformation_matrixes = []
        for i in range(c_struct.facial_transformation_matrixes_count):
            matrix_c = c_struct.facial_transformation_matrixes[i]
            facial_transformation_matrixes.append(matrix_c.to_numpy())

        return cls(face_landmarks, face_blendshapes, facial_transformation_matrixes)

    Result = face_landmarker.FaceLandmarkerResult
    Result._unpatched_from_ctypes = Result.from_ctypes  # diagnostic / unpatch hook
    Result.from_ctypes = classmethod(_patched_from_ctypes)
    _PATCH_APPLIED = True
    logger.info("Mediapipe lazy-landmark patch applied (struct_size=%d).", struct_size)
    return True
