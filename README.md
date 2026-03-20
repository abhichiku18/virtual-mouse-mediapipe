# 🖱️ AI Virtual Mouse — Hand Gesture Control System

> **Control your computer mouse using just your hand gestures — no physical mouse required. Built with Computer Vision and MediaPipe.**

---

## 🎥 Demo Video

[![Watch Demo](https://img.shields.io/badge/Watch-Demo_Video-red?style=for-the-badge&logo=youtube)](ADD_YOUR_VIDEO_LINK_HERE)

> 📌 Replace `ADD_YOUR_VIDEO_LINK_HERE` with your YouTube / Google Drive link.

---

## 📌 Project Overview

This project implements a **real-time AI-powered virtual mouse** that tracks hand movements through a webcam and translates them into mouse actions using computer vision techniques.

- 👆 **Move** your index finger → mouse cursor moves
- 🤏 **Pinch** index finger + thumb → mouse click

No hardware needed beyond a standard webcam.

---

## ✨ Features

- 🎯 Real-time hand landmark detection (21 keypoints)
- 🖱️ Smooth cursor movement with jitter reduction
- 👆 Single-finger mouse control
- 🤏 Pinch-to-click gesture recognition
- 🪞 Mirrored camera view for natural interaction
- ⚡ Low-latency per-frame processing
- 🔴🟢 Visual dot markers on thumb and index finger

---

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ Move index finger | Moves mouse cursor |
| 🤏 Pinch (index + thumb < 40px apart) | Left mouse click |

---

## 🧠 How It Works

```
Webcam Feed
     ↓
Flip & Convert to RGB
     ↓
MediaPipe Hand Detection (21 Landmarks)
     ↓
Extract Index Finger Tip (ID=8) + Thumb Tip (ID=4)
     ↓
Map finger coordinates → Screen coordinates
     ↓
Smoothening filter applied (factor = 7)
     ↓
PyAutoGUI moves cursor
     ↓
If distance(index, thumb) < 40px → Click
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Camera_Feed-5C3EE8?style=flat&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-orange?style=flat)
![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-Mouse_Control-green?style=flat)

---

## 🔬 Key Concepts Used

| Concept | Implementation |
|---------|----------------|
| **Hand Landmark Detection** | MediaPipe detects 21 hand keypoints in real-time |
| **Coordinate Mapping** | Finger position mapped from camera frame to screen resolution |
| **Smoothening Filter** | Moving average applied to remove cursor jitter |
| **Gesture Recognition** | Euclidean distance between fingertips for click detection |
| **Real-time Processing** | Per-frame processing loop using OpenCV |

---

## 📐 Hand Landmarks Used

```
Index Finger Tip  →  Landmark ID = 8   (Green dot 🟢)
Thumb Tip         →  Landmark ID = 4   (Red dot 🔴)
```

> MediaPipe provides 21 hand landmarks total — this project uses 2 key points for full mouse control.

---

## 📁 Project Structure

```
Virtual-Mouse/
│
├── virtual_mouse.py      # Main source code
└── README.md             # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Virtual-Mouse.git
cd Virtual-Mouse
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe pyautogui
```

### 3️⃣ Run the Project

```bash
python virtual_mouse.py
```

### 4️⃣ Exit

Press **`Q`** to quit the application.

---

## 📦 requirements

```
opencv-python
mediapipe
pyautogui
```

---

## 🚧 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Cursor jittering due to hand tremors | Applied smoothening factor (÷7) using moving average |
| Accidental clicks from minor finger overlap | Set minimum distance threshold of 40px for click trigger |
| Camera coordinates differ from screen coordinates | Mapped frame dimensions to screen resolution via `pyautogui.size()` |
| Left-right mirror confusion | Used `cv2.flip(frame, 1)` for natural mirrored view |

---

## 🎯 Future Improvements

- 🖱️ Right-click gesture (3-finger pinch)
- 📜 Scroll gesture (two-finger swipe)
- 🪟 Multi-monitor support
- ✋ Two-hand gesture support
- 🎛️ Configurable sensitivity settings
- 💻 GUI control panel for gesture mapping

---

## 🧠 One-Line Summary (For Resume / Interview)

> *"Built a real-time AI virtual mouse using OpenCV and MediaPipe that maps hand landmark coordinates to screen positions, with a smoothening filter for stable cursor control and Euclidean distance-based pinch detection for click events."*

---

## ⚠️ Requirements

- Python 3.7+
- Webcam (built-in or external)
- Good lighting for accurate hand detection

---

## 🧠 Skills Demonstrated

| Skill | Detail |
|-------|--------|
| **Computer Vision** | Real-time webcam processing with OpenCV |
| **AI / ML Integration** | MediaPipe hand tracking model |
| **Math / Geometry** | Euclidean distance for gesture detection |
| **System Automation** | OS-level mouse control via PyAutoGUI |
| **Signal Processing** | Smoothening filter to reduce noise |

---

## 👨‍💻 Author

**Abhinav Chaudhary**

[![GitHub](https://img.shields.io/badge/GitHub-abhichiku18-black?style=flat&logo=github)](https://github.com/abhichiku18)

---

## ⭐ Support

If you found this project cool, give it a **⭐ star** on GitHub!
