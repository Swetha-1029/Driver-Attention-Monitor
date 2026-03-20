"""
app.py
======
Flask web server for Driver Attention Monitoring System.
Streams webcam + detection data to the browser in real time.

Run:
    python app.py
Then open: http://localhost:5000
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import json
from flask import Flask, Response, render_template, jsonify

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    EAR_THRESHOLD, MAR_THRESHOLD, HEAD_TILT_THRESH,
    EAR_CONSEC_FRAMES, MAR_CONSEC_FRAMES,
    HEAD_DISTRACTION_SECS, NO_FACE_ALERT_SECS,
    RECOVERY_RATE, RECOVERY_DELAY_SECS,
    EAR_PENALTY, MAR_PENALTY, HEAD_PENALTY,
    ATTENTION_ALERT_THRESHOLD, BRIGHT_MIN, BRIGHT_MAX
)
from scipy.spatial import distance as dist

# ── Exact working detection functions from attention_monitor.py ──
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

def eye_aspect_ratio(lm, eye_indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.3

def mouth_aspect_ratio(lm, w, h):
    top    = (lm[MOUTH_TOP].x    * w, lm[MOUTH_TOP].y    * h)
    bottom = (lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h)
    left   = (lm[MOUTH_LEFT].x   * w, lm[MOUTH_LEFT].y   * h)
    right  = (lm[MOUTH_RIGHT].x  * w, lm[MOUTH_RIGHT].y  * h)
    vertical   = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0

def get_head_pose(lm):
    nose_x    = lm[NOSE_TIP].x;  nose_y    = lm[NOSE_TIP].y
    leye_x    = lm[L_EYE_OUT].x; reye_x    = lm[R_EYE_OUT].x
    leye_y    = lm[L_EYE_OUT].y; reye_y    = lm[R_EYE_OUT].y
    chin_y    = lm[CHIN].y
    eye_mid_x = (leye_x + reye_x) / 2.0
    eye_mid_y = (leye_y + reye_y) / 2.0
    face_h    = abs(chin_y - eye_mid_y) if abs(chin_y - eye_mid_y) != 0 else 1
    return nose_x - eye_mid_x, (nose_y - eye_mid_y) / face_h

def compute_all(lm, w, h):
    l = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
    r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
    return (l+r)/2.0, mouth_aspect_ratio(lm, w, h), *get_head_pose(lm)

app   = Flask(__name__)

# ─── Shared state (thread-safe via lock) ─────────────────────
lock  = threading.Lock()
state = {
    "running":       False,
    "ear":           0.0,
    "mar":           0.0,
    "tilt_x":        0.0,
    "tilt_y":        0.0,
    "eye_alert":     False,
    "yawn_alert":    False,
    "head_alert":    False,
    "no_face":       False,
    "attention":     100.0,
    "light_warn":    None,
    "alert_log":     [],      # [{time, type}]
    "calibrated":    False,
    "calib_progress": 0,
    "session_start": None,
}

latest_frame = None   # JPEG bytes of latest annotated frame


# ─── Detection thread ────────────────────────────────────────
def detection_loop():
    global latest_frame

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.65, min_tracking_confidence=0.65
    )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # ── Calibration ──────────────────────────────────────────
    ear_samples, mar_samples = [], []
    calib_start = time.time()
    CALIB_SECS  = 4

    while time.time() - calib_start < CALIB_SECS:
        ret, frame = cap.read()
        if not ret:
            continue
        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elapsed = time.time() - calib_start
        prog    = int((elapsed / CALIB_SECS) * 100)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            l  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            r  = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear_samples.append((l + r) / 2.0)
            mar_samples.append(mouth_aspect_ratio(lm, w, h))

        with lock:
            state["calib_progress"] = prog

        # Send calibration frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "CALIBRATING...", (w//2-160, h//2-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,255), 2)
        cv2.putText(frame, f"Eyes open | Mouth closed  {CALIB_SECS - elapsed:.1f}s",
                    (w//2-220, h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_frame = buf.tobytes()
        time.sleep(0.03)

    ear_thresh = EAR_THRESHOLD
    mar_thresh = MAR_THRESHOLD
    if len(ear_samples) > 5:
        ear_thresh = np.mean(ear_samples) * 0.72  # lower for specs wearers
        mar_thresh = np.mean(mar_samples) + 0.30

    with lock:
        state["calibrated"]    = True
        state["running"]       = True
        state["session_start"] = time.time()

    # ── Main loop ─────────────────────────────────────────────
    ear_counter      = 0
    mar_counter      = 0
    head_turn_start  = 0.0
    head_was_turned  = False
    no_face_start    = 0.0
    face_was_missing = False
    last_alert_time  = 0.0
    attention        = 100.0

    while True:
        with lock:
            if not state["running"]:
                break

        ret, frame = cap.read()
        if not ret:
            continue

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        ear = mar = tilt_x = tilt_y = 0.0
        eye_alert = yawn_alert = head_alert = False

        # Lighting
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        light_warn = "TOO DARK" if brightness < BRIGHT_MIN else ("TOO BRIGHT" if brightness > BRIGHT_MAX else None)

        if results.multi_face_landmarks:
            lm               = results.multi_face_landmarks[0].landmark
            face_was_missing = False
            no_face_start    = 0.0

            ear, mar, tilt_x, tilt_y = compute_all(lm, w, h)

            ear_counter = ear_counter + 1 if ear < ear_thresh else max(0, ear_counter - 3)
            eye_alert   = ear_counter >= 12  # reduced from 20 for specs wearers

            if mar > mar_thresh:
                mar_counter += 1
            else:
                mar_counter = max(0, mar_counter - 6)
            yawn_alert = mar_counter >= MAR_CONSEC_FRAMES

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

            any_alert = eye_alert or yawn_alert or head_alert
            if eye_alert:
                attention -= EAR_PENALTY
                last_alert_time = time.time()
            if yawn_alert:
                attention -= MAR_PENALTY
                last_alert_time = time.time()
            if head_alert:
                attention -= HEAD_PENALTY
                last_alert_time = time.time()
            if not any_alert:
                if time.time() - last_alert_time >= RECOVERY_DELAY_SECS:
                    attention = min(100.0, attention + RECOVERY_RATE)
        else:
            if not face_was_missing:
                no_face_start    = time.time()
                face_was_missing = True
            if time.time() - no_face_start >= NO_FACE_ALERT_SECS:
                attention = max(0.0, attention - HEAD_PENALTY)
                last_alert_time = time.time()

        attention = max(0.0, min(100.0, attention))

        # Build alert log entry
        alert_type = None
        if eye_alert:   alert_type = "EYES"
        elif yawn_alert: alert_type = "YAWN"
        elif head_alert: alert_type = "HEAD"
        elif face_was_missing and (time.time() - no_face_start) >= NO_FACE_ALERT_SECS:
            alert_type = "NO FACE"

        with lock:
            state.update({
                "ear": round(ear, 3), "mar": round(mar, 3),
                "tilt_x": round(tilt_x, 3), "tilt_y": round(tilt_y, 3),
                "eye_alert": eye_alert, "yawn_alert": yawn_alert,
                "head_alert": head_alert,
                "no_face": face_was_missing and (time.time() - no_face_start) >= NO_FACE_ALERT_SECS,
                "attention": round(attention, 1),
                "light_warn": light_warn,
            })
            if alert_type:
                log = state["alert_log"]
                if not log or log[-1]["type"] != alert_type or time.time() - log[-1]["ts"] > 5:
                    log.append({"type": alert_type, "ts": time.time(),
                                "score": round(attention, 1)})
                    if len(log) > 50:
                        log.pop(0)

        # Annotate frame minimally
        color = (0,200,80) if attention >= 75 else (0,180,255) if attention >= 50 else (0,50,220)
        cv2.putText(frame, f"Score: {int(attention)}%", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if eye_alert:
            cv2.putText(frame, "EYES DROWSY", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,220), 2)
        if yawn_alert:
            cv2.putText(frame, "YAWNING", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,220), 2)
        if head_alert:
            cv2.putText(frame, "HEAD TURNED", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,220), 2)
        if attention < ATTENTION_ALERT_THRESHOLD:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,180), 4)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_frame = buf.tobytes()
        time.sleep(0.03)

    cap.release()


# ─── Flask routes ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    while True:
        if latest_frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.03)


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data')
def data():
    with lock:
        s = dict(state)
        s["alert_log"] = [
            {"type": e["type"], "score": e["score"],
             "time": time.strftime('%H:%M:%S', time.localtime(e["ts"]))}
            for e in s["alert_log"][-10:]
        ]
        if s["session_start"]:
            elapsed = int(time.time() - s["session_start"])
            s["session_duration"] = f"{elapsed//60:02d}:{elapsed%60:02d}"
        else:
            s["session_duration"] = "00:00"
    return jsonify(s)


@app.route('/stop', methods=['POST'])
def stop():
    with lock:
        state["running"] = False
    return jsonify({"ok": True})


if __name__ == '__main__':
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("\n[OK] Open http://localhost:5000 in your browser\n")
    app.run(debug=False, threaded=True)
