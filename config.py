"""
config.py
=========
All configurable settings for the Driver Attention Monitoring System.
Edit this file to tune the system for different users or environments.
"""

# ─────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────
CALIBRATION_SECONDS = 4        # Duration of startup calibration

# ─────────────────────────────────────────────
#  DETECTION THRESHOLDS
# (These are defaults — overridden by calibration)
# ─────────────────────────────────────────────
EAR_THRESHOLD           = 0.25  # Eye Aspect Ratio below this = eyes closing
MAR_THRESHOLD           = 0.75  # Mouth Aspect Ratio above this = yawning
HEAD_TILT_THRESH        = 0.04  # Nose deviation — side turn sensitivity
HEAD_NOD_THRESH         = 0.65  # Nose-to-eye ratio — forward nod sensitivity

# ─────────────────────────────────────────────
#  CONSECUTIVE FRAME COUNTS
# ─────────────────────────────────────────────
EAR_CONSEC_FRAMES       = 20   # Frames of eye closure before alert
MAR_CONSEC_FRAMES       = 45   # Increased — avoids false triggers from lip licking/talking

# ─────────────────────────────────────────────
#  TIME-BASED DETECTION
# ─────────────────────────────────────────────
HEAD_DISTRACTION_SECS   = 2.0  # Seconds of head turn before alert (ignores quick glances)
NO_FACE_ALERT_SECS      = 3.0  # Seconds face missing before alert

# ─────────────────────────────────────────────
#  ATTENTION SCORE
# ─────────────────────────────────────────────
ATTENTION_ALERT_THRESHOLD = 50  # Score below this triggers alarm
EAR_PENALTY             = 1.5   # Score drop per frame when eyes closing
MAR_PENALTY             = 1.0   # Score drop per frame when yawning
HEAD_PENALTY            = 1.0   # Score drop per frame when head turned
RECOVERY_RATE           = 0.15  # Score recovery per frame — slow, fatigue doesnt vanish instantly

# ─────────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────────
BEEP_COOLDOWN           = 2.5   # Minimum seconds between beeps
RECOVERY_DELAY_SECS     = 3.0   # Seconds of clear alerts before score starts recovering

# ─────────────────────────────────────────────
#  LIGHTING
# ─────────────────────────────────────────────
BRIGHT_MIN              = 40    # Below this = too dark
BRIGHT_MAX              = 210   # Above this = too bright

# ─────────────────────────────────────────────
#  WEBCAM
# ─────────────────────────────────────────────
WEBCAM_INDEX            = 0     # 0 = default webcam, 1 = external camera
FRAME_WIDTH             = 1280
FRAME_HEIGHT            = 720
