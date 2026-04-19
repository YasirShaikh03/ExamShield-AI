<h1 align="center">🎓 AI Exam Proctoring System — Elite Edition</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-green?logo=google" />
  <img src="https://img.shields.io/badge/YOLOv4--tiny-Device%20Detection-red" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <b>Real-time AI-powered exam proctoring with 478-point face landmarks, 3D head pose, iris tracking & electronic device detection.</b>
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/YasirShaikh03"><b>Yasir Shaikh</b></a>
</p>

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🧠 Face Landmarks | MediaPipe 0.10+ · 478-point model |
| 👁️ Iris Gaze Tracking | Left / Right / Up / Down / Center |
| 🗣️ 3D Head Pose | Pitch · Yaw · Roll via solvePnP |
| 📱 Device Detection | YOLOv4-tiny · Phone, Laptop, Keyboard, Remote, Mouse |
| 📚 Soft Warnings | Book/Notes, Smartwatch in frame |
| 👥 Multi-Face Tracking | Intruder alert + new person detection |
| 👀 Blink Rate Monitor | Flags abnormal blink frequency |
| 📏 Distance Check | Too close / too far from camera |
| 🔊 Alarm | Audio alert on every cheat event |
| 📄 Session Reports | JSON report + CSV event log per session |
| 📸 Screenshots | Press `S` to capture flagged moments |

---

## 🎥 Detection Priority

```
0. 📵 Electronic Device Detected   🚨  (HIGHEST)
1. 👥 Multiple Faces               🚨
2. 🚫 Face Not Detected            🚨
3. ◀▶ Head Left / Right            ❌
4. 👁️  Eye Gaze Left / Right        ❌
5. ▲▼ Head Tilt Up / Face Down     ❌
6. ⚠️  Blink / Distance / Fidget    ⚠
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YasirShaikh03/ai-exam-proctoring.git
cd ai-exam-proctoring
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python proctoring.py
```

> **First run** will automatically download:
> - `face_landmarker.task` (~30 MB) — MediaPipe face model
> - `yolov4-tiny.weights` (~24 MB) — YOLO device detection model
> - `yolov4-tiny.cfg` and `coco.names`

---

## 📦 Requirements

```
opencv-python>=4.5
mediapipe>=0.10
numpy>=1.21
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⌨️ Controls

| Key | Action |
|---|---|
| `ESC` | End session & save report |
| `D` | Toggle iris debug overlay |
| `R` | Reset cheat counter |
| `S` | Save screenshot |

---

## 📊 Behavior Rules

| Behavior | Classification |
|---|---|
| Eye Center / Up / Down | ✅ Normal |
| Small natural head movement | ✅ Normal |
| Eye Gaze Left or Right | ❌ CHEAT |
| Head turned Left or Right (Yaw > 20°) | ❌ CHEAT |
| Head tilted Up (Pitch > 20°) | ❌ CHEAT |
| Face tilted Down (Pitch < -20°) | ❌ CHEAT |
| Eyes closed > 1.5s | ⚠ WARN |
| Blink rate < 6 or > 40 /min | ⚠ WARN |
| Too close or too far from camera | ⚠ WARN |
| Phone / Laptop / Keyboard detected | ❌ CHEAT |
| Book / Notes / Smartwatch detected | ⚠ WARN |
| Multiple faces in frame | ❌ CHEAT |
| Face not detected | ❌ CHEAT |

---

## 📁 Project Structure

```
ai-exam-proctoring/
├── proctoring.py           # Main script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── logs/                   # Session CSV & JSON reports (auto-created)
├── screenshots/            # Captured screenshots (auto-created)
└── .github/
    └── workflows/
        └── lint.yml        # CI lint check
```

> Model files (`face_landmarker.task`, `yolov4-tiny.*`) are downloaded automatically on first run and are excluded from git via `.gitignore`.

---

## 📝 Output Files

After each session, two files are saved in `logs/`:

- **`events_<session_id>.csv`** — Timestamped log of every flagged event
- **`report_<session_id>.json`** — Full session summary with risk level and stats

**Risk levels:** `Clean ✓` · `Low ⚠` · `Medium ⚠⚠` · `High 🚨`

---

## 🧩 Architecture

```
Camera Feed
    │
    ├─► MediaPipe FaceLandmarker (478 pts)
    │       ├─ Iris Gaze Estimator
    │       ├─ EAR Blink Detector
    │       ├─ 3D Head Pose (solvePnP)
    │       └─ Face Distance & Movement Tracker
    │
    ├─► Haar Cascade → Multi-Face Tracker
    │
    └─► YOLOv4-tiny → Electronic Device Detector
            └─ Phone · Laptop · Keyboard · Remote · Mouse · Book
```

---

## ⚙️ Configuration

You can tune detection sensitivity by editing constants at the top of `proctoring.py`:

| Constant | Default | Effect |
|---|---|---|
| `EAR_THRESH` | `0.20` | Blink sensitivity |
| `CONF_THRESH` | `0.40` | YOLO detection confidence |
| `SKIP_FRAMES` | `5` | Run YOLO every N frames |
| `INPUT_SIZE` | `(416,416)` | YOLO input size (smaller = faster) |
| Yaw threshold | `±20°` | Head left/right sensitivity |
| Pitch threshold | `±20°` | Head tilt sensitivity |

---

## 🖥️ Platform Support

| Platform | Camera | Audio Alarm |
|---|---|---|
| Windows | ✅ DirectShow / MSMF | ✅ `winsound` |
| macOS | ✅ AVFoundation | ✅ `afplay` |
| Linux | ✅ V4L2 | ✅ `aplay` |

---

## 📜 License

MIT License © 2024 [Yasir Shaikh](https://github.com/YasirShaikh03)

---

<p align="center">
  <a href="https://github.com/YasirShaikh03">
    <img src="https://img.shields.io/badge/GitHub-YasirShaikh03-181717?logo=github&style=for-the-badge" />
  </a>
</p>
