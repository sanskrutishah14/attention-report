"""
attention_classifier.py
------------------------
Classifies each FaceIndicators object into one of five attention states
using a rule-based scoring system calibrated on behavioral research thresholds.

States (ordered by engagement level):
    FOCUSED      — Eyes open, looking forward, head stable
    LISTENING    — Slight head tilt or side gaze, still attentive
    UNFOCUSED    — Head turned, wandering gaze, reduced engagement
    DISTRACTED   — Significant head rotation, looking away from board
    SLEEPY       — Low EAR (eyes closing), slow movement, drooping head

All thresholds are configurable via the ClassifierConfig dataclass.
No identity or biometric data is stored — only the state label per frame.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import numpy as np

from engagement_detector import FaceIndicators


# ---------------------------------------------------------------------------
# Attention State Enum
# ---------------------------------------------------------------------------

class AttentionState(Enum):
    FOCUSED     = "Focused"
    LISTENING   = "Listening"
    UNFOCUSED   = "Unfocused"
    DISTRACTED  = "Distracted"
    SLEEPY      = "Sleepy"

# Numeric engagement scores for metric calculations
STATE_SCORES: Dict[AttentionState, float] = {
    AttentionState.FOCUSED:    1.00,
    AttentionState.LISTENING:  0.75,
    AttentionState.UNFOCUSED:  0.45,
    AttentionState.DISTRACTED: 0.20,
    AttentionState.SLEEPY:     0.05,
}

# Colors for OpenCV overlay (BGR)
STATE_COLORS: Dict[AttentionState, tuple] = {
    AttentionState.FOCUSED:    (50,  220, 50 ),
    AttentionState.LISTENING:  (50,  180, 220),
    AttentionState.UNFOCUSED:  (30,  140, 255),
    AttentionState.DISTRACTED: (0,   80,  255),
    AttentionState.SLEEPY:     (100, 50,  200),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """Tunable thresholds for the rule-based classifier."""

    # EAR thresholds
    ear_sleepy_threshold:  float = 0.20   # Below this = eyes nearly closed
    ear_blink_threshold:   float = 0.25   # Transient blink (ignore in sleepy check)

    # MAR threshold for yawning
    mar_yawn_threshold: float = 0.55

    # Head pose thresholds (degrees)
    yaw_forward_max:    float = 18.0   # Within ±18° yaw = looking at board
    yaw_listening_max:  float = 30.0   # ±18–30° = slight turn, still listening
    yaw_distracted_min: float = 40.0   # Beyond ±40° yaw = clearly distracted

    pitch_forward_max:  float = 20.0   # Looking slightly down (notes) is OK
    pitch_sleepy_min:   float = 25.0   # Head drooping down = sleepy signal
    pitch_up_max:       float = 20.0   # Looking up at board is fine

    # Gaze thresholds (normalized, 0 = straight)
    gaze_focused_max:   float = 0.12   # Small offset = still focused
    gaze_wander_max:    float = 0.22   # Medium offset = listening

    # Movement thresholds (pixels)
    movement_restless:  float = 8.0    # High movement = distracted/restless
    movement_frozen:    float = 0.5    # Zero movement + low EAR = sleepy

    # Smoothing window for temporal state smoothing
    smoothing_window: int = 7


# ---------------------------------------------------------------------------
# Per-face classification result
# ---------------------------------------------------------------------------

@dataclass
class FaceClassification:
    face_id: int
    state: AttentionState
    score: float            # 0.0 – 1.0 engagement score
    confidence: float       # Classifier confidence in the state
    reason: str             # Human-readable primary reason


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class AttentionClassifier:
    """
    Rule-based attention classifier with temporal smoothing.

    Each call to `classify_frame` returns a FaceClassification per face.
    History buffers enable smoothing over the last N frames per face_id.
    """

    def __init__(self, config: ClassifierConfig = None):
        self.cfg = config or ClassifierConfig()
        # face_id -> deque of recent AttentionState values for smoothing
        self._history: Dict[int, List[AttentionState]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_frame(self, faces: List[FaceIndicators]) -> List[FaceClassification]:
        """Classify all faces in a single frame."""
        results = []
        for fi in faces:
            raw_state, confidence, reason = self._classify_single(fi)
            smoothed = self._smooth(fi.face_id, raw_state)
            results.append(FaceClassification(
                face_id=fi.face_id,
                state=smoothed,
                score=STATE_SCORES[smoothed],
                confidence=confidence,
                reason=reason,
            ))
        return results

    def classify_indicators(self, fi: FaceIndicators) -> FaceClassification:
        """Classify a single FaceIndicators object (no smoothing)."""
        state, confidence, reason = self._classify_single(fi)
        return FaceClassification(fi.face_id, state, STATE_SCORES[state], confidence, reason)

    # ------------------------------------------------------------------
    # Core rule engine
    # ------------------------------------------------------------------

    def _classify_single(self, fi: FaceIndicators):
        """
        Multi-condition scoring system. Each condition votes with a weight.
        Returns (AttentionState, confidence, reason).
        """
        cfg = self.cfg
        scores = {s: 0.0 for s in AttentionState}
        primary_reason = "nominal"

        abs_yaw   = abs(fi.yaw)
        abs_pitch = abs(fi.pitch)
        abs_gaze  = abs(fi.gaze_x)

        # -------------------------------------------------------------------
        # SLEEPY signals
        # -------------------------------------------------------------------
        sleepy_votes = 0

        if fi.ear < cfg.ear_sleepy_threshold:
            scores[AttentionState.SLEEPY] += 4.0
            sleepy_votes += 1
            primary_reason = f"eyes closing (EAR={fi.ear:.2f})"

        if fi.pitch > cfg.pitch_sleepy_min:            # Head drooping forward
            scores[AttentionState.SLEEPY] += 2.5
            sleepy_votes += 1
            if sleepy_votes >= 2:
                primary_reason = "head drooping and eyes closing"

        if fi.mar > cfg.mar_yawn_threshold:
            scores[AttentionState.SLEEPY] += 2.0
            primary_reason = f"yawning (MAR={fi.mar:.2f})"

        if fi.movement_delta < cfg.movement_frozen and fi.ear < cfg.ear_blink_threshold:
            scores[AttentionState.SLEEPY] += 1.5

        # -------------------------------------------------------------------
        # DISTRACTED signals
        # -------------------------------------------------------------------
        if abs_yaw > cfg.yaw_distracted_min:
            scores[AttentionState.DISTRACTED] += 4.0
            primary_reason = f"head turned away (yaw={fi.yaw:+.0f}°)"

        if abs_gaze > cfg.gaze_wander_max and abs_yaw > cfg.yaw_listening_max:
            scores[AttentionState.DISTRACTED] += 2.5

        if fi.movement_delta > cfg.movement_restless and abs_yaw > cfg.yaw_forward_max:
            scores[AttentionState.DISTRACTED] += 1.5
            primary_reason = f"restless movement + looking away"

        # -------------------------------------------------------------------
        # UNFOCUSED signals (mild disengagement)
        # -------------------------------------------------------------------
        if cfg.yaw_forward_max < abs_yaw <= cfg.yaw_distracted_min:
            scores[AttentionState.UNFOCUSED] += 2.5

        if abs_gaze > cfg.gaze_focused_max:
            scores[AttentionState.UNFOCUSED] += 1.0

        if cfg.ear_blink_threshold < fi.ear < 0.28:    # Somewhat sleepy
            scores[AttentionState.UNFOCUSED] += 1.0

        if fi.movement_delta > cfg.movement_restless:
            scores[AttentionState.UNFOCUSED] += 0.5

        # -------------------------------------------------------------------
        # LISTENING signals (attentive but slightly off-axis)
        # -------------------------------------------------------------------
        if cfg.yaw_forward_max < abs_yaw <= cfg.yaw_listening_max:
            scores[AttentionState.LISTENING] += 2.0

        if abs_gaze <= cfg.gaze_wander_max:
            scores[AttentionState.LISTENING] += 1.0

        if fi.ear >= cfg.ear_blink_threshold:
            scores[AttentionState.LISTENING] += 0.5

        # -------------------------------------------------------------------
        # FOCUSED signals
        # -------------------------------------------------------------------
        if abs_yaw <= cfg.yaw_forward_max:
            scores[AttentionState.FOCUSED] += 3.0

        if abs_gaze <= cfg.gaze_focused_max:
            scores[AttentionState.FOCUSED] += 2.0

        if fi.ear >= cfg.ear_blink_threshold:
            scores[AttentionState.FOCUSED] += 1.5

        if abs_pitch <= cfg.pitch_forward_max:
            scores[AttentionState.FOCUSED] += 1.0

        if fi.movement_delta < cfg.movement_restless:
            scores[AttentionState.FOCUSED] += 0.5

        # -------------------------------------------------------------------
        # Pick winner
        # -------------------------------------------------------------------
        best_state = max(scores, key=lambda s: scores[s])
        total = sum(scores.values()) or 1.0
        confidence = scores[best_state] / total

        if primary_reason == "nominal":
            primary_reason = self._default_reason(fi, best_state)

        return best_state, float(confidence), primary_reason

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def _smooth(self, face_id: int, raw_state: AttentionState) -> AttentionState:
        """Mode of the last N states for this face."""
        if face_id not in self._history:
            self._history[face_id] = []
        buf = self._history[face_id]
        buf.append(raw_state)
        if len(buf) > self.cfg.smoothing_window:
            buf.pop(0)

        # Weighted mode: recent frames have higher weight
        weights = np.linspace(0.5, 1.0, len(buf))
        tally = {s: 0.0 for s in AttentionState}
        for w, s in zip(weights, buf):
            tally[s] += w
        return max(tally, key=lambda s: tally[s])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_reason(fi: FaceIndicators, state: AttentionState) -> str:
        if state == AttentionState.FOCUSED:
            return f"head forward (yaw={fi.yaw:+.0f}°), eyes open (EAR={fi.ear:.2f})"
        elif state == AttentionState.LISTENING:
            return f"slight head turn (yaw={fi.yaw:+.0f}°)"
        elif state == AttentionState.UNFOCUSED:
            return f"partial disengagement (yaw={fi.yaw:+.0f}°, gaze_x={fi.gaze_x:.2f})"
        elif state == AttentionState.DISTRACTED:
            return f"significant head rotation (yaw={fi.yaw:+.0f}°)"
        else:
            return f"low EAR ({fi.ear:.2f})"


# ---------------------------------------------------------------------------
# Utility: batch classify a list of FrameResults
# ---------------------------------------------------------------------------

def classify_session(frame_results, config: ClassifierConfig = None):
    """
    Classify all frames in a session.

    Args:
        frame_results: Iterable of FrameResult from EngagementDetector.
        config: Optional ClassifierConfig.

    Yields:
        (FrameResult, List[FaceClassification]) tuples.
    """
    clf = AttentionClassifier(config)
    for fr in frame_results:
        classifications = clf.classify_frame(fr.faces)
        yield fr, classifications
