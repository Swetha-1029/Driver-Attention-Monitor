## 👁️ Driver Attention Monitor — Real-Time Drowsiness Detection
A real-time driver attention monitoring system built with Python, OpenCV, MediaPipe and Flask.
Detects eye closure, yawning, and head pose — combining them into a live Attention Score (0–100%) with a web dashboard and audio alerts.

💡 Built during my internship in 2024.
---

## Project Structure

```
driver_attention_project/
│
├── main.py                  ← Entry point — run this
├── config.py                ← All settings and thresholds
├── requirements.txt         ← Python dependencies
│
├── modules/
│   ├── detector.py          ← EAR, MAR, Head Pose calculations
│   ├── score_engine.py      ← Attention Score logic
│   ├── alert.py             ← Audio beep alert
│   └── ui.py                ← HUD overlay drawing
│
└── logs/                    ← Session logs (future)
```

---

## Features

| Feature | Description |
|---|---|
| EAR Detection | Detects eye closure using Eye Aspect Ratio |
| MAR Detection | Detects yawning using Mouth Aspect Ratio |
| Head Pose | Detects forward nod and side turns |
| Smart Head Detection | Quick glances ignored — only sustained turns (2s+) trigger alert |
| Face Missing Alert | Alerts when driver's face disappears for 3+ seconds |
| Attention Score | Live 0–100% score combining all signals |
| Startup Calibration | Adapts thresholds to each user's face (works with specs) |
| Lighting Warning | Warns when lighting is too dark or too bright |
| Audio Alert | Beep sound when score drops below 50% |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

Press **Q** to quit.

---

## How the Attention Score Works

```
Score starts at 100%

Eyes closing  (EAR < threshold)  → -1.5 per frame
Yawning       (MAR > threshold)  → -1.0 per frame
Head turned   (> 2 seconds)      → -1.0 per frame
Face missing  (> 3 seconds)      → -1.0 per frame
All clear                        → +0.5 recovery per frame

Score < 50% → RED alert + audio beep
```

---

## Unique Contributions

1. **Per-user calibration** — measures baseline EAR/MAR at startup for personalised thresholds
2. **Multi-signal Attention Score** — combines 3 signals into one live percentage
3. **Time-based head pose** — distinguishes mirror checks from distraction
4. **Face disappearance detection** — catches when driver looks completely away

---

## Tech Stack
- Python 3.x
- OpenCV
- MediaPipe Face Mesh
- SciPy
- NumPy
- Pygame

---

## Future Scope
- Session log export to CSV
- Streamlit web dashboard
- Raspberry Pi deployment with dashboard camera
- SMS alert to emergency contact

## 👩‍💻 Author
Swetha
