"""
main.py
=======
Real-Time Driver Attention Monitoring System
Entry point — run this file to start the system.

Usage:
    python main.py
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import sys

from config import (
    CALIBRATION_SECONDS, WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    EAR_THRESHOLD, MAR_THRESHOLD, HEAD_TILT_THRESH,
    EAR_CONSEC_FRAMES, MAR_CONSEC_FRAMES,
    HEAD_DISTRACTION_SECS, NO_FACE_ALERT_SECS,
    BRIGHT_MIN, BRIGHT_MAX
)
from modules.detector  import compute_all
from modules.score_engine import AttentionScoreEngine
from modules.alert     import AudioAlert
from modules.ui        import draw_hud


# ─────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────
def run_calibration(cap, face_mesh):
    from modules.detector import eye_aspect_ratio, mouth_aspect_ratio
    from modules.detector import LEFT_EYE, RIGHT_EYE

    print("\n[CALIBRATION] Keep face relaxed — eyes open, mouth closed.")
    print(f"[CALIBRATION] Collecting for {CALIBRATION_SECONDS} seconds...\n")

    ear_samples, mar_samples = [], []
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb)
        elapsed   = time.time() - start
        remaining = max(0, CALIBRATION_SECONDS - elapsed)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "CALIBRATING...", (w//2 - 160, h//2 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3)
        cv2.putText(frame, "Eyes OPEN  |  Mouth CLOSED", (w//2 - 200, h//2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Please wait: {remaining:.1f}s", (w//2 - 130, h//2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 150), 2)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            l  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            r  = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear_samples.append((l + r) / 2.0)
            mar_samples.append(mouth_aspect_ratio(lm, w, h))
            cv2.putText(frame, "Face detected!", (w//2 - 90, h//2 + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
        else:
            cv2.putText(frame, "No face — move closer", (w//2 - 160, h//2 + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

        cv2.imshow("Driver Attention Monitor", frame)
        cv2.waitKey(1)

        if elapsed >= CALIBRATION_SECONDS:
            break

    ear_thresh = EAR_THRESHOLD
    mar_thresh = MAR_THRESHOLD

    if len(ear_samples) > 10:
        avg_ear    = np.mean(ear_samples)
        avg_mar    = np.mean(mar_samples)
        ear_thresh = avg_ear * 0.78
        mar_thresh = avg_mar + 0.30
        print(f"[CALIBRATION] Done!")
        print(f"  EAR baseline: {avg_ear:.3f}  → Drowsy threshold: {ear_thresh:.3f}")
        print(f"  MAR baseline: {avg_mar:.3f}  → Yawn  threshold: {mar_thresh:.3f}\n")
    else:
        print("[CALIBRATION] Not enough samples — using defaults.\n")

    return ear_thresh, mar_thresh


# ─────────────────────────────────────────────
#  LIGHTING CHECK
# ─────────────────────────────────────────────
def check_lighting(frame):
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if brightness < BRIGHT_MIN:
        return "TOO DARK",   (0, 80, 255)
    elif brightness > BRIGHT_MAX:
        return "TOO BRIGHT", (0, 200, 255)
    return None, None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Real-Time Driver Attention Monitor v2.0")
    print("=" * 50)

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )

    # Modules
    audio  = AudioAlert()
    engine = AttentionScoreEngine()

    # Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print("[OK] Webcam opened.")

    # Calibration
    ear_thresh, mar_thresh = run_calibration(cap, face_mesh)

    # State
    ear_counter      = 0
    mar_counter      = 0
    head_turn_start  = 0.0
    head_was_turned  = False
    no_face_start    = 0.0
    face_was_missing = False

    print("[OK] Monitoring started. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        ear = mar = tilt_x = tilt_y = 0.0
        eye_alert = yawn_alert = head_alert = False
        light_warn, light_color = check_lighting(frame)

        if results.multi_face_landmarks:
            lm               = results.multi_face_landmarks[0].landmark
            face_was_missing = False
            no_face_start    = 0.0

            ear, mar, tilt_x, tilt_y = compute_all(lm, w, h)

            # EAR counter
            ear_counter = ear_counter + 1 if ear < ear_thresh else max(0, ear_counter - 3)
            eye_alert   = ear_counter >= EAR_CONSEC_FRAMES

            # MAR counter — decays fast to avoid lip/talking false alarms
            if mar > mar_thresh:
                mar_counter += 1
            else:
                mar_counter = max(0, mar_counter - 6)
            yawn_alert = mar_counter >= MAR_CONSEC_FRAMES

            # Head — time based
            head_turned = abs(tilt_x) > HEAD_TILT_THRESH or tilt_y > 0.65
            if head_turned:
                if not head_was_turned:
                    head_turn_start = time.time()
                head_was_turned = True
                if time.time() - head_turn_start >= HEAD_DISTRACTION_SECS:
                    head_alert = True
            else:
                head_was_turned = False
                head_turn_start = 0.0

            engine.update(eye_alert, yawn_alert, head_alert)

        else:
            if not face_was_missing:
                no_face_start    = time.time()
                face_was_missing = True
            if time.time() - no_face_start >= NO_FACE_ALERT_SECS:
                engine.penalise_no_face()

        # Alert
        if engine.is_critical:
            audio.play()

        # Durations for UI
        head_duration    = (time.time() - head_turn_start) if head_was_turned  else 0.0
        no_face_duration = (time.time() - no_face_start)   if face_was_missing else 0.0

        frame = draw_hud(frame, ear, mar, tilt_x, tilt_y, engine,
                         eye_alert, yawn_alert, head_alert, engine.is_critical,
                         light_warn, light_color, head_duration, no_face_duration)

        cv2.imshow("Driver Attention Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
