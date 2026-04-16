"""
engagement_detector.py
-----------------------
Real-time face detection and behavioral indicator extraction using
MediaPipe FaceMesh + OpenCV. Processes webcam or video input frame-by-frame.

Extracted indicators per face:
  - Head pose (yaw, pitch, roll)
  - Gaze direction (estimated from iris landmarks)
  - Eye Aspect Ratio (EAR) — for drowsiness detection
  - Mouth Aspect Ratio (MAR) — for yawning
  - Movement delta (head displacement between frames)
  - Bounding box for overlay rendering
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FaceIndicators:
    """All behavioral indicators extracted for a single detected face."""
    face_id: int                        # Temporary per-frame index (no identity)
    bbox: tuple                         # (x, y, w, h) in pixels
    yaw: float = 0.0                    # Head turn left/right (degrees)
    pitch: float = 0.0                  # Head tilt up/down (degrees)
    roll: float = 0.0                   # Head tilt sideways (degrees)
    gaze_x: float = 0.0                 # Horizontal gaze offset (normalized)
    gaze_y: float = 0.0                 # Vertical gaze offset (normalized)
    ear: float = 0.3                    # Eye Aspect Ratio (lower = more closed)
    mar: float = 0.0                    # Mouth Aspect Ratio (higher = yawning)
    movement_delta: float = 0.0         # Euclidean distance of nose tip vs prev frame
    confidence: float = 1.0            # MediaPipe detection confidence


@dataclass
class FrameResult:
    """All extracted data for a single video frame."""
    frame_index: int
    timestamp_sec: float
    faces: list = field(default_factory=list)   # List[FaceIndicators]
    annotated_frame: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe FaceMesh 468-point model)
# ---------------------------------------------------------------------------

# Eyes (right/left from subject's perspective)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]

# Iris centres
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS  = [473, 474, 475, 476, 477]

# Mouth corners & top/bottom lip
MOUTH_INDICES = [61, 291, 13, 14, 17, 0]

# Nose tip and key 3-D reference points for head pose
NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE_L   = 263
RIGHT_EYE_R  = 33
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    """Compute EAR from 6 eye landmarks.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * frame_w, lm.y * frame_h]))

    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * h)


def _mouth_aspect_ratio(landmarks, frame_w, frame_h):
    """Compute MAR — ratio of vertical mouth opening to horizontal width."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    top    = pt(13)
    bottom = pt(14)
    left   = pt(61)
    right  = pt(291)
    vert   = np.linalg.norm(top - bottom)
    horiz  = np.linalg.norm(left - right)
    if horiz < 1e-6:
        return 0.0
    return vert / horiz


def _estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Approximate yaw/pitch/roll using a simplified PnP approach with
    6 canonical 3-D face points and their 2-D MediaPipe projections.
    Returns (yaw, pitch, roll) in degrees.
    """
    # 3-D model points (generic face, metric units don't matter — ratios do)
    model_3d = np.array([
        [0.0,    0.0,    0.0  ],   # Nose tip
        [0.0,   -63.6,  -12.5],   # Chin
        [-43.3,  32.7,  -26.0],   # Left eye left corner
        [43.3,   32.7,  -26.0],   # Right eye right corner
        [-28.9, -28.9,  -24.1],   # Left mouth corner
        [28.9,  -28.9,  -24.1],   # Right mouth corner
    ], dtype=np.float64)

    key_indices = [NOSE_TIP, CHIN, LEFT_EYE_L, RIGHT_EYE_R, LEFT_MOUTH, RIGHT_MOUTH]

    def lm2d(idx):
        lm = landmarks[idx]
        return [lm.x * frame_w, lm.y * frame_h]

    image_2d = np.array([lm2d(i) for i in key_indices], dtype=np.float64)

    focal  = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_3d, image_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix to Euler angles
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0

    return float(yaw), float(pitch), float(roll)


def _estimate_gaze(landmarks, frame_w, frame_h):
    """
    Estimate normalized gaze offset from iris position relative to eye corner.
    Returns (gaze_x, gaze_y) where 0,0 = looking straight ahead.
    Positive x = looking right, positive y = looking down.
    """
    def iris_center(iris_idx_list):
        xs = [landmarks[i].x for i in iris_idx_list]
        ys = [landmarks[i].y for i in iris_idx_list]
        return np.array([np.mean(xs), np.mean(ys)])

    try:
        r_iris = iris_center(RIGHT_IRIS)
        l_iris = iris_center(LEFT_IRIS)

        r_corner_l = np.array([landmarks[33].x,  landmarks[33].y])
        r_corner_r = np.array([landmarks[133].x, landmarks[133].y])
        l_corner_l = np.array([landmarks[362].x, landmarks[362].y])
        l_corner_r = np.array([landmarks[263].x, landmarks[263].y])

        def gaze_offset(iris, corner_l, corner_r):
            eye_width = np.linalg.norm(corner_r - corner_l)
            if eye_width < 1e-6:
                return 0.0, 0.0
            rel = iris - corner_l
            norm_x = rel[0] / eye_width - 0.5   # 0 = centre
            norm_y = rel[1] / eye_width
            return float(norm_x), float(norm_y)

        rx, ry = gaze_offset(r_iris, r_corner_l, r_corner_r)
        lx, ly = gaze_offset(l_iris, l_corner_l, l_corner_r)

        return (rx + lx) / 2, (ry + ly) / 2

    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class EngagementDetector:
    """
    Wraps MediaPipe FaceMesh to process video frames and extract
    per-face behavioral indicators.

    Usage:
        detector = EngagementDetector()
        for result in detector.process_video("lecture.mp4"):
            ...   # FrameResult with face indicators

        # Or live webcam:
        detector.run_webcam(on_frame_callback=my_func)
    """

    def __init__(
        self,
        max_faces: int = 30,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,   # Required for iris landmarks
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing   = mp.solutions.drawing_utils
        self.mp_styles    = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Track nose-tip positions across frames for movement delta
        self._prev_nose_positions: dict = {}   # face_id -> (x, y)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_index: int = 0,
                      timestamp_sec: float = 0.0, draw: bool = True) -> FrameResult:
        """
        Process a single BGR frame.
        Returns a FrameResult with extracted FaceIndicators for each face found.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy() if draw else None
        face_list = []

        if not results.multi_face_landmarks:
            return FrameResult(frame_index, timestamp_sec, [], annotated)

        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            lms = face_landmarks.landmark

            # Bounding box
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            bbox = (x1, y1, x2 - x1, y2 - y1)

            # Indicators
            ear_r = _eye_aspect_ratio(lms, RIGHT_EYE_INDICES, w, h)
            ear_l = _eye_aspect_ratio(lms, LEFT_EYE_INDICES,  w, h)
            ear   = (ear_r + ear_l) / 2.0

            mar  = _mouth_aspect_ratio(lms, w, h)
            yaw, pitch, roll = _estimate_head_pose(lms, w, h)
            gaze_x, gaze_y   = _estimate_gaze(lms, w, h)

            # Movement delta (nose tip vs previous frame)
            nose_x = lms[NOSE_TIP].x * w
            nose_y = lms[NOSE_TIP].y * h
            prev   = self._prev_nose_positions.get(face_id)
            if prev is not None:
                movement_delta = float(np.linalg.norm(
                    np.array([nose_x, nose_y]) - np.array(prev)
                ))
            else:
                movement_delta = 0.0
            self._prev_nose_positions[face_id] = (nose_x, nose_y)

            fi = FaceIndicators(
                face_id=face_id, bbox=bbox,
                yaw=yaw, pitch=pitch, roll=roll,
                gaze_x=gaze_x, gaze_y=gaze_y,
                ear=ear, mar=mar,
                movement_delta=movement_delta,
            )
            face_list.append(fi)

            if draw and annotated is not None:
                self._draw_face(annotated, fi, lms, w, h)

        return FrameResult(frame_index, timestamp_sec, face_list, annotated)

    # ------------------------------------------------------------------
    # Video file processing
    # ------------------------------------------------------------------

    def process_video(self, video_path: str, sample_every_n_frames: int = 5,
                      show_preview: bool = False):
        """
        Generator that yields FrameResult for each sampled frame of a video file.

        Args:
            video_path: Path to .mp4 / .avi / etc.
            sample_every_n_frames: Process 1 out of every N frames (speed vs accuracy trade-off).
            show_preview: Show annotated frames in an OpenCV window.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % sample_every_n_frames == 0:
                    ts = frame_index / fps
                    result = self.process_frame(frame, frame_index, ts, draw=show_preview)

                    if show_preview and result.annotated_frame is not None:
                        cv2.imshow("Classroom Analyzer", result.annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    yield result

                frame_index += 1
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Live webcam
    # ------------------------------------------------------------------

    def run_webcam(self, camera_index: int = 0,
                   on_frame_callback=None,
                   sample_every_n_frames: int = 3,
                   show_preview: bool = True):
        """
        Run real-time analysis on webcam feed.

        Args:
            camera_index: OpenCV camera index (usually 0).
            on_frame_callback: Called with each FrameResult. If returns False, stops.
            sample_every_n_frames: Analyse every Nth frame.
            show_preview: Show live annotated window.
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            raise IOError(f"Cannot open camera index {camera_index}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % sample_every_n_frames == 0:
                    ts = frame_index / fps
                    result = self.process_frame(frame, frame_index, ts, draw=show_preview)

                    if on_frame_callback is not None:
                        keep_going = on_frame_callback(result)
                        if keep_going is False:
                            break

                    if show_preview and result.annotated_frame is not None:
                        cv2.imshow("Classroom Analyzer — Live", result.annotated_frame)

                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                frame_index += 1
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_face(self, frame, fi: FaceIndicators, lms, w, h):
        """Draw bounding box, pose axes, and EAR/MAR info on frame."""
        x, y, fw, fh = fi.bbox

        # Color by rough focus (will be refined by classifier later)
        color = (0, 220, 0)   # green default
        if abs(fi.yaw) > 30 or abs(fi.pitch) > 25:
            color = (0, 165, 255)   # orange — looking away
        if fi.ear < 0.18:
            color = (0, 0, 220)     # red — eyes closed

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

        # Overlay text
        info = f"Y:{fi.yaw:+.0f} P:{fi.pitch:+.0f} EAR:{fi.ear:.2f}"
        cv2.putText(frame, info, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Draw head pose axis arrow (yaw)
        cx, cy = x + fw // 2, y + fh // 2
        ax = int(cx + np.sin(np.radians(fi.yaw)) * 40)
        ay = int(cy - np.sin(np.radians(fi.pitch)) * 40)
        cv2.arrowedLine(frame, (cx, cy), (ax, ay), (255, 255, 0), 2, tipLength=0.3)
