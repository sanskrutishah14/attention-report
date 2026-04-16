"""
metrics_engine.py
------------------
Aggregates per-frame face classifications into session-level metrics.

Computed metrics:
  ┌─────────────────────────────┬────────────────────────────────────────────────────────────┐
  │ Metric                      │ Definition                                                 │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Class Attention Score (CAS) │ Mean engagement score across all frames & faces (0–100)   │
  │ Focus Stability Index (FSI) │ 1 – (std dev of per-frame avg score) — consistency        │
  │ Disengagement Duration (DD) │ % of session where class avg engagement < threshold       │
  │ Participation Equity (PES)  │ 1 – (Gini coefficient of per-student attention scores)    │
  │ State Distribution          │ % time in each AttentionState                             │
  │ Attention Timeline          │ Per-minute averaged engagement score                      │
  │ Peak Distraction Windows    │ Time windows with worst engagement (for teacher review)   │
  └─────────────────────────────┴────────────────────────────────────────────────────────────┘

All metrics are class-level — no individual student is profiled.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

from attention_classifier import AttentionState, STATE_SCORES, FaceClassification


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    """Aggregated engagement metrics for a single frame."""
    frame_index: int
    timestamp_sec: float
    num_faces: int
    avg_score: float                           # 0–1 mean engagement
    state_counts: Dict[str, int]               # state_name -> count
    dominant_state: str


@dataclass
class SessionMetrics:
    """Full session-level engagement summary."""
    session_id: str
    duration_sec: float
    total_frames_analyzed: int
    avg_students_detected: float

    # Core KPIs
    class_attention_score: float               # 0–100
    focus_stability_index: float               # 0–1
    disengagement_duration_pct: float          # 0–100
    participation_equity_score: float          # 0–1

    # State breakdown
    state_distribution: Dict[str, float]       # state_name -> % of frame-detections

    # Timeline (per minute)
    attention_timeline: List[Dict]             # [{minute, avg_score, dominant_state}]

    # Worst engagement windows (for report)
    peak_distraction_windows: List[Dict]       # [{start_sec, end_sec, avg_score}]

    # Raw frame-level data for charts
    frame_metrics: List[FrameMetrics] = field(default_factory=list)

    # Engagement level classification
    @property
    def engagement_level(self) -> str:
        if self.class_attention_score >= 70:
            return "High"
        elif self.class_attention_score >= 45:
            return "Moderate"
        else:
            return "Low"


# ---------------------------------------------------------------------------
# Metrics Engine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """
    Accumulates (FrameResult, [FaceClassification]) pairs and computes
    all session metrics when finalize() is called.
    """

    def __init__(
        self,
        session_id: str = "session_01",
        disengagement_threshold: float = 0.40,   # Below this = disengaged frame
        distraction_window_min_dur: float = 30.0  # Seconds for "peak distraction window"
    ):
        self.session_id = session_id
        self.disengagement_threshold = disengagement_threshold
        self.distraction_window_min_dur = distraction_window_min_dur

        self._frame_data: List[Tuple] = []   # (frame_result, [FaceClassification])

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add_frame(self, frame_result, classifications: List[FaceClassification]):
        """Add one frame's data. Call this for every frame in the session."""
        self._frame_data.append((frame_result, classifications))

    def add_frames_batch(self, frame_clf_pairs):
        """Add an iterable of (frame_result, classifications) pairs."""
        for fr, clf in frame_clf_pairs:
            self.add_frame(fr, clf)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> SessionMetrics:
        """Compute all session metrics from accumulated frame data."""
        if not self._frame_data:
            return self._empty_metrics()

        frame_metrics_list = self._compute_frame_metrics()
        duration_sec = self._frame_data[-1][0].timestamp_sec if self._frame_data else 0.0

        # ---- Core KPIs ------------------------------------------------
        all_scores = [fm.avg_score for fm in frame_metrics_list if fm.num_faces > 0]
        if not all_scores:
            all_scores = [0.0]

        cas = float(np.mean(all_scores)) * 100.0
        fsi = max(0.0, 1.0 - float(np.std(all_scores)))
        dd  = (sum(1 for s in all_scores if s < self.disengagement_threshold)
               / len(all_scores)) * 100.0

        pes = self._participation_equity()

        # ---- State distribution ----------------------------------------
        state_dist = self._state_distribution()

        # ---- Attention timeline (per minute) ---------------------------
        timeline = self._build_timeline(frame_metrics_list)

        # ---- Peak distraction windows ----------------------------------
        windows = self._peak_distraction_windows(frame_metrics_list)

        avg_faces = float(np.mean([fm.num_faces for fm in frame_metrics_list])) if frame_metrics_list else 0.0

        return SessionMetrics(
            session_id=self.session_id,
            duration_sec=duration_sec,
            total_frames_analyzed=len(frame_metrics_list),
            avg_students_detected=avg_faces,
            class_attention_score=round(cas, 1),
            focus_stability_index=round(fsi, 3),
            disengagement_duration_pct=round(dd, 1),
            participation_equity_score=round(pes, 3),
            state_distribution=state_dist,
            attention_timeline=timeline,
            peak_distraction_windows=windows,
            frame_metrics=frame_metrics_list,
        )

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    def _compute_frame_metrics(self) -> List[FrameMetrics]:
        results = []
        for fr, clfs in self._frame_data:
            if not clfs:
                fm = FrameMetrics(fr.frame_index, fr.timestamp_sec, 0, 0.0, {}, "No faces")
                results.append(fm)
                continue

            scores = [c.score for c in clfs]
            avg    = float(np.mean(scores))

            state_counts: Dict[str, int] = {}
            for c in clfs:
                name = c.state.value
                state_counts[name] = state_counts.get(name, 0) + 1

            dominant = max(state_counts, key=lambda s: state_counts[s])

            fm = FrameMetrics(
                frame_index=fr.frame_index,
                timestamp_sec=fr.timestamp_sec,
                num_faces=len(clfs),
                avg_score=avg,
                state_counts=state_counts,
                dominant_state=dominant,
            )
            results.append(fm)
        return results

    def _state_distribution(self) -> Dict[str, float]:
        """% of all face-frame detections in each state."""
        totals: Dict[str, int] = {s.value: 0 for s in AttentionState}
        grand_total = 0
        for _, clfs in self._frame_data:
            for c in clfs:
                totals[c.state.value] += 1
                grand_total += 1
        if grand_total == 0:
            return {k: 0.0 for k in totals}
        return {k: round(v / grand_total * 100, 1) for k, v in totals.items()}

    def _participation_equity(self) -> float:
        """
        Participation Equity Score = 1 − Gini(per-face cumulative scores).
        High score (→1) = all students have similar engagement.
        Low score (→0) = highly uneven participation.
        """
        # Accumulate total score per face_id across all frames
        face_totals: Dict[int, List[float]] = {}
        for _, clfs in self._frame_data:
            for c in clfs:
                if c.face_id not in face_totals:
                    face_totals[c.face_id] = []
                face_totals[c.face_id].append(c.score)

        if len(face_totals) < 2:
            return 1.0   # Only 1 face = trivially equitable

        means = np.array([np.mean(v) for v in face_totals.values()])
        gini  = self._gini(means)
        return float(round(1.0 - gini, 3))

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        """Compute Gini coefficient of a 1-D array."""
        arr = np.sort(np.abs(arr))
        n   = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        cumsum = np.cumsum(arr)
        return float((2 * cumsum.sum() - (n + 1) * arr.sum()) / (n * arr.sum()))

    def _build_timeline(self, frame_metrics: List[FrameMetrics]) -> List[Dict]:
        """Average engagement per minute for the timeline chart."""
        buckets: Dict[int, List[float]] = {}
        bucket_states: Dict[int, List[str]] = {}

        for fm in frame_metrics:
            minute = int(fm.timestamp_sec // 60)
            if minute not in buckets:
                buckets[minute] = []
                bucket_states[minute] = []
            if fm.num_faces > 0:
                buckets[minute].append(fm.avg_score)
                bucket_states[minute].append(fm.dominant_state)

        timeline = []
        for minute in sorted(buckets.keys()):
            scores = buckets[minute]
            states = bucket_states[minute]
            avg    = float(np.mean(scores)) if scores else 0.0
            dom    = max(set(states), key=states.count) if states else "Unknown"
            timeline.append({
                "minute": minute,
                "avg_score": round(avg * 100, 1),
                "dominant_state": dom,
            })
        return timeline

    def _peak_distraction_windows(self, frame_metrics: List[FrameMetrics]) -> List[Dict]:
        """
        Find contiguous windows where avg_score < disengagement_threshold.
        Returns top-3 worst windows (lowest average score, min duration met).
        """
        windows = []
        in_window = False
        win_start = 0.0
        win_frames = []

        for fm in frame_metrics:
            if fm.num_faces == 0:
                continue
            is_low = fm.avg_score < self.disengagement_threshold

            if is_low and not in_window:
                in_window = True
                win_start = fm.timestamp_sec
                win_frames = [fm.avg_score]
            elif is_low and in_window:
                win_frames.append(fm.avg_score)
            elif not is_low and in_window:
                duration = fm.timestamp_sec - win_start
                if duration >= self.distraction_window_min_dur:
                    windows.append({
                        "start_sec": round(win_start, 1),
                        "end_sec": round(fm.timestamp_sec, 1),
                        "duration_sec": round(duration, 1),
                        "avg_score": round(float(np.mean(win_frames)) * 100, 1),
                    })
                in_window = False
                win_frames = []

        # Sort by avg_score ascending (worst first), take top 3
        windows.sort(key=lambda w: w["avg_score"])
        return windows[:3]

    def _empty_metrics(self) -> SessionMetrics:
        return SessionMetrics(
            session_id=self.session_id,
            duration_sec=0.0,
            total_frames_analyzed=0,
            avg_students_detected=0.0,
            class_attention_score=0.0,
            focus_stability_index=0.0,
            disengagement_duration_pct=0.0,
            participation_equity_score=0.0,
            state_distribution={s.value: 0.0 for s in AttentionState},
            attention_timeline=[],
            peak_distraction_windows=[],
        )
