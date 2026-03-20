"""
modules/alert.py
================
Handles audio alerts using pygame.
Generates a beep sound when attention score drops below threshold.
"""

import pygame
import numpy as np
import time
from config import BEEP_COOLDOWN


class AudioAlert:
    def __init__(self):
        self.sound     = None
        self.available = False
        self.last_beep = 0.0
        self._init()

    def _init(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            sr   = 44100
            dur  = 0.35
            n    = int(sr * dur)
            t    = np.linspace(0, dur, n, False)
            mono = (np.sin(2 * np.pi * 900 * t) * 28000).astype(np.int16)
            stereo     = np.column_stack((mono, mono))
            self.sound = pygame.sndarray.make_sound(stereo)
            self.available = True
            print("[OK] Audio initialized.")
        except Exception as e:
            print(f"[WARN] Audio unavailable: {e}")

    def play(self):
        """Play beep if cooldown has passed."""
        now = time.time()
        if self.available and (now - self.last_beep) > BEEP_COOLDOWN:
            self.sound.play()
            self.last_beep = now
