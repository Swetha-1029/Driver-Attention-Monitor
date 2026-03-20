"""
modules/detector.py
===================
Face landmark detection and metric calculation.
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head Pose (tilt_x, tilt_y)
"""

from scipy.spatial import distance as dist
import numpy as np

# ─────────────────────────────────────────────
#  LANDMARK INDICES (MediaPipe Face Mesh)
# ─────────────────────────────────────────────
LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
NOSE_TIP     = 4
L_EYE_OUT    = 33
R_EYE_OUT    = 263
CHIN         = 152


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Calculate Eye Aspect Ratio (EAR).
    EAR drops when eyes close.
    Normal open eye: ~0.30   Closed eye: ~0.20
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.3


def mouth_aspect_ratio(landmarks, w, h):
    """
    Calculate Mouth Aspect Ratio (MAR).
    MAR rises when mouth opens wide (yawn).
    Normal closed: ~0.05   Wide yawn: ~0.5+
    """
    top    = (landmarks[MOUTH_TOP].x    * w, landmarks[MOUTH_TOP].y    * h)
    bottom = (landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h)
    left   = (landmarks[MOUTH_LEFT].x   * w, landmarks[MOUTH_LEFT].y   * h)
    right  = (landmarks[MOUTH_RIGHT].x  * w, landmarks[MOUTH_RIGHT].y  * h)
    vertical   = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0


def get_head_pose(landmarks):
    """
    Simple stable head pose using nose vs eye-chin geometry.
    No solvePnP — works reliably on all faces including specs wearers.

    Returns:
        tilt_x: left/right turn  (threshold ~0.04)
        tilt_y: forward nod      (threshold ~0.65)
    """
    nose_x    = landmarks[NOSE_TIP].x
    nose_y    = landmarks[NOSE_TIP].y
    leye_x    = landmarks[L_EYE_OUT].x
    reye_x    = landmarks[R_EYE_OUT].x
    leye_y    = landmarks[L_EYE_OUT].y
    reye_y    = landmarks[R_EYE_OUT].y
    chin_y    = landmarks[CHIN].y

    eye_mid_x = (leye_x + reye_x) / 2.0
    eye_mid_y = (leye_y + reye_y) / 2.0
    face_h    = abs(chin_y - eye_mid_y) if abs(chin_y - eye_mid_y) != 0 else 1

    tilt_x = nose_x - eye_mid_x
    tilt_y = (nose_y - eye_mid_y) / face_h
    return tilt_x, tilt_y


def compute_all(landmarks, w, h):
    """
    Compute EAR, MAR, and head pose in one call.
    Returns: (ear, mar, tilt_x, tilt_y)
    """
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear       = (left_ear + right_ear) / 2.0
    mar       = mouth_aspect_ratio(landmarks, w, h)
    tilt_x, tilt_y = get_head_pose(landmarks)
    return ear, mar, tilt_x, tilt_y
