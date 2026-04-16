# AI Classroom Engagement & Learning Equity Analyzer

Anonymous, privacy-preserving classroom engagement analysis using computer vision + generative AI.

---

## Architecture

```
main.py
  │
  ├── engagement_detector.py   ← MediaPipe FaceMesh: head pose, gaze, EAR, MAR
  ├── attention_classifier.py  ← Rule-based classifier → Focused / Listening / Unfocused / Distracted / Sleepy
  ├── metrics_engine.py        ← Aggregates frames → CAS, FSI, DD, PES
  ├── report_generator.py      ← natural language session report
  └── visualizer.py            ← Matplotlib dashboard charts
```

**No facial recognition. No identity storage. All analysis is class-level.**

---

## Setup

### 1. Python 3.9+

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Anthropic API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS/Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
```

### 3. GPU (optional but recommended)

MediaPipe and OpenCV will use CUDA automatically if available.
For GPU-accelerated MediaPipe, ensure CUDA 11.x+ and `nvidia-cuda-toolkit` are installed.

---

## Usage

### Demo Mode (no camera required)
Tests the full pipeline with synthetic 10-student, 30-minute session data:
```bash
python main.py demo
```
Outputs: `reports/demo_session_001_dashboard.png` and `reports/demo_session_report.md`

---

### Webcam Mode (live classroom)
```bash
python main.py webcam
python main.py webcam --teacher "Ms. Sharma" --subject "Physics" --camera 0
```
- Press **Q** in the preview window to stop and generate the report.
- Use `--no-preview` to run headlessly (useful on server setups).

---

### Video File Mode
```bash
python main.py video lecture.mp4
python main.py video lecture.mp4 --session-id "week3_physics" --teacher "Dr. Khan"
```

---

### All Options
```
python main.py [webcam|video|demo] [video_path]
               [--camera N]         Webcam index (default: 0)
               [--session-id STR]   Session label for filenames
               [--teacher STR]      Teacher name for report
               [--subject STR]      Subject/topic taught
               [--sample N]         Analyze every Nth frame (default: 3, lower = more accurate)
               [--no-preview]       Disable OpenCV window
```

---

## Output Files

All outputs are saved to the `reports/` directory:

| File | Description |
|------|-------------|
| `{session_id}_metrics.json` | Raw session metrics (KPIs, timeline, state distribution) |
| `{session_id}_dashboard.png` | 3-panel visual dashboard |
| `{session_id}_report.md` | Full AI-generated teaching report |

---

## Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| **Class Attention Score (CAS)** | 0–100 | Average engagement level across the session |
| **Focus Stability Index (FSI)** | 0–1 | How consistent attention was (1 = very stable) |
| **Disengagement Duration (DD)** | 0–100% | % of session below 40% engagement threshold |
| **Participation Equity Score (PES)** | 0–1 | How evenly distributed attention was (1 = equal) |

---

## Attention States

| State | Color | Score | Description |
|-------|-------|-------|-------------|
| Focused | 🟢 Green | 1.00 | Head forward, eyes open, gaze on board |
| Listening | 🔵 Blue | 0.75 | Slight head turn, still attentive |
| Unfocused | 🟠 Orange | 0.45 | Wandering gaze, mild disengagement |
| Distracted | 🔴 Red | 0.20 | Significant head rotation, looking away |
| Sleepy | 🟣 Purple | 0.05 | Low EAR, head drooping, minimal movement |

---

## Tuning the Classifier

Edit thresholds in `attention_classifier.py → ClassifierConfig`:

```python
config = ClassifierConfig(
    ear_sleepy_threshold=0.20,     # EAR below this = sleepy
    yaw_forward_max=18.0,          # Degrees: ±18° = looking at board
    yaw_distracted_min=40.0,       # Degrees: >40° = clearly distracted
    smoothing_window=7,            # Frames: temporal smoothing
)
```

---

## Privacy & Ethics

- **No facial recognition** — faces are detected but never identified
- **No data persistence** — no images, biometrics, or face embeddings are stored
- **Class-level only** — all reports aggregate to group metrics
- **Consent required** — always inform students before deployment
- **Not for surveillance** — designed to improve teaching, not monitor individuals

---

## Project Structure

```
classroom_analyzer/
├── main.py                  ← Entry point
├── engagement_detector.py   ← Face/pose detection
├── attention_classifier.py  ← State classification
├── metrics_engine.py        ← KPI computation
├── report_generator.py      ← report
├── visualizer.py            ← Charts & dashboard
├── requirements.txt
├── README.md
└── reports/                 ← Generated output (auto-created)
```
