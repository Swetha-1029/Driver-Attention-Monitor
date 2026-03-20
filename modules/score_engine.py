"""
modules/score_engine.py
=======================
Manages the Driver Attention Score.
- Drops score when alerts fire
- Recovers score when driver is alert
- Tracks alert history
"""

import time
from config import (
    ATTENTION_ALERT_THRESHOLD,
    EAR_PENALTY, MAR_PENALTY, HEAD_PENALTY, RECOVERY_RATE,
    RECOVERY_DELAY_SECS
)


class AttentionScoreEngine:
    def __init__(self):
        self.score           = 100.0
        self.alert_log       = []
        self.last_alert_time = 0.0

    def update(self, eye_alert, yawn_alert, head_alert):
        """Update score based on current alerts."""
        any_alert = eye_alert or yawn_alert or head_alert
        if eye_alert:
            self.score -= EAR_PENALTY
            self.last_alert_time = time.time()
            self.alert_log.append((time.time(), "EYE"))
        if yawn_alert:
            self.score -= MAR_PENALTY
            self.last_alert_time = time.time()
            self.alert_log.append((time.time(), "YAWN"))
        if head_alert:
            self.score -= HEAD_PENALTY
            self.last_alert_time = time.time()
            self.alert_log.append((time.time(), "HEAD"))
        if not any_alert:
            time_since_alert = time.time() - self.last_alert_time
            if time_since_alert >= RECOVERY_DELAY_SECS:
                self.score = min(100.0, self.score + RECOVERY_RATE)

        self.score = max(0.0, min(100.0, self.score))

    def penalise_no_face(self):
        """Penalise score when face is not detected for too long."""
        self.score = max(0.0, self.score - HEAD_PENALTY)
        self.alert_log.append((time.time(), "NO_FACE"))

    @property
    def is_critical(self):
        return self.score <= ATTENTION_ALERT_THRESHOLD

    @property
    def grade(self):
        if self.score >= 75:
            return "ALERT",  (0, 210, 70)
        elif self.score >= 50:
            return "TIRED",  (0, 190, 255)
        else:
            return "DANGER", (0, 50, 255)

    def reset(self):
        self.score     = 100.0
        self.alert_log = []
