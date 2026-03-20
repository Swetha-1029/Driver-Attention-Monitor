"""
modules/ui.py
=============
Draws the HUD overlay on the webcam frame.
- Left panel: metrics, status, score bar
- Alert flash when score is critical
- Lighting and face detection warnings
"""

import cv2
from config import EAR_THRESHOLD, MAR_THRESHOLD, HEAD_TILT_THRESH, NO_FACE_ALERT_SECS


def draw_hud(frame, ear, mar, tilt_x, tilt_y, score_engine,
             eye_alert, yawn_alert, head_alert, alert_active,
             light_warn, light_color, head_duration=0.0, no_face_duration=0.0):

    h, w = frame.shape[:2]

    # ── Side panel ────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (315, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Title ─────────────────────────────────────────────────
    cv2.putText(frame, "DRIVER ATTENTION", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "MONITOR", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.line(frame, (10, 68), (305, 68), (80, 80, 80), 1)

    # ── Raw metrics ───────────────────────────────────────────
    metrics = [
        ("EAR",    f"{ear:.3f}",    ear < EAR_THRESHOLD),
        ("MAR",    f"{mar:.3f}",    mar > MAR_THRESHOLD),
        ("H-SIDE", f"{tilt_x:.3f}", abs(tilt_x) > HEAD_TILT_THRESH),
        ("H-NOD",  f"{tilt_y:.3f}", tilt_y > 0.65),
    ]
    y = 92
    for label, val, warn in metrics:
        col = (0, 100, 255) if warn else (180, 180, 180)
        cv2.putText(frame, f"{label}:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)
        cv2.putText(frame, val, (110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
        y += 26

    cv2.line(frame, (10, y + 2), (305, y + 2), (80, 80, 80), 1)
    y += 18

    # ── Status indicators ─────────────────────────────────────
    if head_duration > 0 and not head_alert:
        head_status = f"{head_duration:.1f}s"
    elif head_alert:
        head_status = "ALERT"
    else:
        head_status = "OK"

    statuses = [
        ("EYES", "DROWSY" if eye_alert  else "OK",  eye_alert),
        ("YAWN", "YES"    if yawn_alert else "NO",  yawn_alert),
        ("HEAD", head_status,                        head_alert),
    ]
    for label, status, warn in statuses:
        col = (0, 70, 255) if warn else (0, 210, 90)
        cv2.putText(frame, f"{label}:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        cv2.putText(frame, status, (110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += 28

    cv2.line(frame, (10, y + 2), (305, y + 2), (80, 80, 80), 1)
    y += 18

    # ── Lighting warning ──────────────────────────────────────
    if light_warn:
        cv2.putText(frame, f"LIGHT: {light_warn}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, light_color, 2)
        y += 26

    # ── Face not detected warning ─────────────────────────────
    if no_face_duration >= NO_FACE_ALERT_SECS:
        cv2.putText(frame, "FACE: NOT DETECTED", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 50, 255), 2)
        y += 26
    elif no_face_duration > 0:
        cv2.putText(frame, f"FACE: {no_face_duration:.1f}s missing", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 255), 2)
        y += 26

    # ── Attention score ───────────────────────────────────────
    score_int        = int(score_engine.score)
    grade, bar_col   = score_engine.grade
    lbl_col          = bar_col

    cv2.putText(frame, "ATTENTION SCORE", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
    y += 28
    cv2.putText(frame, f"{score_int}%  {grade}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, lbl_col, 2)
    y += 24

    bx, bw, bh = 10, 285, 20
    cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, y), (bx + int(bw * score_int / 100), y + bh), bar_col, -1)
    cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), (100, 100, 100), 1)

    # ── Alert flash ───────────────────────────────────────────
    if alert_active:
        al = frame.copy()
        cv2.rectangle(al, (0, 0), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(al, 0.18, frame, 0.82, 0, frame)
        cv2.putText(frame, "!! ATTENTION ALERT !!", (w // 2 - 215, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 50, 255), 3)
        cv2.putText(frame, "Please focus on the road!", (w // 2 - 190, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Press Q to quit", (w - 185, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    return frame
