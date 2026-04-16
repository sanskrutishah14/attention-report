"""
main.py
--------
AI Classroom Engagement and Learning Equity Analyzer
Entry point — orchestrates the full pipeline.

Modes:
  webcam   — Live analysis from webcam (press Q to stop and generate report)
  video    — Analyze a recorded video file
  demo     — Generate a synthetic demo report (no camera needed)

Usage:
  python main.py webcam
  python main.py video path/to/lecture.mp4
  python main.py demo
  python main.py webcam --teacher "Ms. Sharma" --subject "Physics"
  python main.py video lecture.mp4 --session-id "physics_w3_d2" --no-preview

Environment Variables:
  ANTHROPIC_API_KEY   — Required for report generation
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional

# ---- Project modules -------------------------------------------------------
from engagement_detector import EngagementDetector
from attention_classifier import AttentionClassifier, ClassifierConfig, classify_session
from metrics_engine import MetricsEngine
from report_generator import ReportGenerator
from visualizer import generate_dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_id(base: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}" if base else f"session_{ts}"


def _ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def _print_banner():
    banner = r"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║   AI Classroom Engagement & Learning Equity Analyzer            ║
 ║   Anonymous · Class-Level · Privacy-Preserving                  ║
 ╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_pipeline(
    frame_source,           # Iterable of FrameResult OR generator
    session_id: str,
    teacher_name: Optional[str] = None,
    subject: Optional[str] = None,
    generate_report: bool = True,
    open_dashboard: bool = False,
):
    """
    Core pipeline: detect → classify → aggregate → report.
    Works for both video files and live webcam frames.
    """
    print(f"\n[Pipeline] Starting session: {session_id}")

    engine = MetricsEngine(session_id=session_id)
    clf    = AttentionClassifier(ClassifierConfig())
    total_frames = 0
    total_faces  = 0
    t_start = time.time()

    for frame_result in frame_source:
        classifications = clf.classify_frame(frame_result.faces)
        engine.add_frame(frame_result, classifications)
        total_frames += 1
        total_faces  += len(frame_result.faces)

        # Progress every 100 frames
        if total_frames % 100 == 0:
            elapsed = time.time() - t_start
            fps_effective = total_frames / elapsed if elapsed > 0 else 0
            print(f"  [Pipeline] Frames processed: {total_frames}  |  "
                  f"Faces detected (total): {total_faces}  |  "
                  f"Effective FPS: {fps_effective:.1f}")

    print(f"\n[Pipeline] Finished. {total_frames} frames analyzed, {total_faces} face-detections.")

    # Compute metrics
    print("[Pipeline] Computing session metrics...")
    metrics = engine.finalize()

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"  Class Attention Score:      {metrics.class_attention_score:.1f}/100")
    print(f"  Focus Stability Index:      {metrics.focus_stability_index:.3f}")
    print(f"  Disengagement Duration:     {metrics.disengagement_duration_pct:.1f}%")
    print(f"  Participation Equity Score: {metrics.participation_equity_score:.3f}")
    print(f"  Overall Level:              {metrics.engagement_level}")
    print(f"{'='*60}\n")

    # Save metrics JSON
    _ensure_reports_dir()
    metrics_path = f"reports/{session_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "session_id": metrics.session_id,
            "duration_sec": metrics.duration_sec,
            "class_attention_score": metrics.class_attention_score,
            "focus_stability_index": metrics.focus_stability_index,
            "disengagement_duration_pct": metrics.disengagement_duration_pct,
            "participation_equity_score": metrics.participation_equity_score,
            "engagement_level": metrics.engagement_level,
            "state_distribution": metrics.state_distribution,
            "attention_timeline": metrics.attention_timeline,
            "peak_distraction_windows": metrics.peak_distraction_windows,
        }, f, indent=2)
    print(f"[Pipeline] Metrics saved: {metrics_path}")

    # Generate dashboard chart
    dashboard_path = generate_dashboard(metrics, output_dir="reports", session_label=session_id)

    # Generate AI report
    if generate_report:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\n[Pipeline] ⚠  ANTHROPIC_API_KEY not set. Skipping AI report generation.")
            print("           Set the env var and re-run to get the full narrative report.\n")
        else:
            try:
                generator = ReportGenerator(api_key=api_key)
                report = generator.generate(
                    metrics,
                    teacher_name=teacher_name,
                    subject=subject,
                )
                report_path = f"reports/{session_id}_report.md"
                report.save(report_path)
                report.print_summary()
            except Exception as e:
                print(f"[Pipeline] Report generation failed: {e}")

    print(f"\n[Pipeline] All outputs saved to: reports/")
    return metrics


# ---------------------------------------------------------------------------
# Mode: Live Webcam
# ---------------------------------------------------------------------------

def run_webcam_mode(
    camera_index: int = 0,
    session_id: Optional[str] = None,
    teacher_name: Optional[str] = None,
    subject: Optional[str] = None,
    show_preview: bool = True,
    sample_every: int = 3,
):
    """Run live webcam analysis. Press Q to stop and generate report."""
    sid = _session_id(session_id or "webcam")

    print(f"\n[Webcam] Camera {camera_index} | Session: {sid}")
    print("[Webcam] Press  Q  in the preview window to stop recording and generate report.\n")

    detector = EngagementDetector(max_faces=30)
    engine   = MetricsEngine(session_id=sid)
    clf      = AttentionClassifier(ClassifierConfig())
    total_frames = 0

    def on_frame(frame_result):
        nonlocal total_frames
        classifications = clf.classify_frame(frame_result.faces)
        engine.add_frame(frame_result, classifications)
        total_frames += 1

        # Draw classification labels on the preview
        if show_preview and frame_result.annotated_frame is not None:
            import cv2
            for c in classifications:
                fi = next((f for f in frame_result.faces if f.face_id == c.face_id), None)
                if fi:
                    x, y, fw, fh = fi.bbox
                    from visualizer import STATE_COLORS_HEX
                    hex_color = STATE_COLORS_HEX.get(c.state.value, "#ffffff")
                    # Convert hex to BGR
                    h = hex_color.lstrip("#")
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    bgr = (b, g, r)
                    cv2.rectangle(frame_result.annotated_frame, (x, y), (x+fw, y+fh), bgr, 2)
                    cv2.putText(frame_result.annotated_frame,
                                c.state.value,
                                (x, y + fh + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2, cv2.LINE_AA)

        return True   # Keep going

    detector.run_webcam(
        camera_index=camera_index,
        on_frame_callback=on_frame,
        sample_every_n_frames=sample_every,
        show_preview=show_preview,
    )

    print(f"\n[Webcam] Recording stopped. {total_frames} frames captured.")
    metrics = engine.finalize()

    _ensure_reports_dir()
    import json
    with open(f"reports/{sid}_metrics.json", "w") as f:
        json.dump({
            "session_id": metrics.session_id,
            "class_attention_score": metrics.class_attention_score,
            "focus_stability_index": metrics.focus_stability_index,
            "disengagement_duration_pct": metrics.disengagement_duration_pct,
            "participation_equity_score": metrics.participation_equity_score,
            "engagement_level": metrics.engagement_level,
            "state_distribution": metrics.state_distribution,
            "attention_timeline": metrics.attention_timeline,
        }, f, indent=2)

    generate_dashboard(metrics, output_dir="reports", session_label=sid)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        generator = ReportGenerator(api_key=api_key)
        report = generator.generate(metrics, teacher_name=teacher_name, subject=subject)
        report.save(f"reports/{sid}_report.md")
        report.print_summary()
    else:
        print("\n[Webcam] Set ANTHROPIC_API_KEY to generate the AI narrative report.")

    print(f"[Webcam] Reports saved to: reports/")


# ---------------------------------------------------------------------------
# Mode: Video File
# ---------------------------------------------------------------------------

def run_video_mode(
    video_path: str,
    session_id: Optional[str] = None,
    teacher_name: Optional[str] = None,
    subject: Optional[str] = None,
    show_preview: bool = False,
    sample_every: int = 5,
):
    """Analyze a recorded video file."""
    if not os.path.isfile(video_path):
        print(f"[Error] Video file not found: {video_path}")
        sys.exit(1)

    sid = _session_id(session_id or "video")
    detector = EngagementDetector(max_faces=30)

    frame_gen = detector.process_video(
        video_path,
        sample_every_n_frames=sample_every,
        show_preview=show_preview,
    )

    run_pipeline(
        frame_source=frame_gen,
        session_id=sid,
        teacher_name=teacher_name,
        subject=subject,
    )


# ---------------------------------------------------------------------------
# Mode: Demo (synthetic data, no camera required)
# ---------------------------------------------------------------------------

def run_demo_mode():
    """
    Generate a full demo report using synthetically simulated session data.
    No camera, GPU, or video file needed.
    """
    import random
    from engagement_detector import FrameResult, FaceIndicators
    from attention_classifier import AttentionClassifier, ClassifierConfig

    print("[Demo] Generating synthetic classroom session (10 students, 30 min)...")
    random.seed(42)

    NUM_STUDENTS   = 10
    SESSION_FRAMES = 1800    # ~30 min at 1 fps equivalent
    FPS_EQUIV      = 1.0

    clf    = AttentionClassifier(ClassifierConfig())
    engine = MetricsEngine(session_id="demo_session_001")

    for fi_idx in range(SESSION_FRAMES):
        ts = fi_idx / FPS_EQUIV
        minute = ts / 60

        # Simulate engagement dip in middle and near the end
        base_engagement = 0.75
        if 10 < minute < 18:    # Mid-session dip
            base_engagement = 0.45
        elif minute > 25:        # End-of-session fatigue
            base_engagement = 0.55

        faces = []
        for student_id in range(NUM_STUDENTS):
            # Each student has slightly different baseline
            student_bias = random.gauss(0, 0.1)
            eff_eng = max(0.1, min(1.0, base_engagement + student_bias))

            # Map engagement to indicator ranges
            if eff_eng > 0.75:
                yaw, pitch, ear, mar, gaze_x = (
                    random.gauss(0, 8),
                    random.gauss(-5, 5),
                    random.gauss(0.30, 0.03),
                    random.gauss(0.1, 0.05),
                    random.gauss(0, 0.05),
                )
            elif eff_eng > 0.45:
                yaw, pitch, ear, mar, gaze_x = (
                    random.gauss(15, 10),
                    random.gauss(5, 8),
                    random.gauss(0.26, 0.04),
                    random.gauss(0.15, 0.07),
                    random.gauss(0.1, 0.08),
                )
            else:
                yaw, pitch, ear, mar, gaze_x = (
                    random.gauss(35, 15),
                    random.gauss(20, 10),
                    random.gauss(0.18, 0.05),
                    random.gauss(0.3, 0.1),
                    random.gauss(0.2, 0.1),
                )

            fi = FaceIndicators(
                face_id=student_id,
                bbox=(student_id * 80, 100, 60, 80),
                yaw=float(yaw), pitch=float(pitch), roll=0.0,
                gaze_x=float(gaze_x), gaze_y=0.0,
                ear=max(0.05, float(ear)),
                mar=max(0.0, float(mar)),
                movement_delta=abs(random.gauss(2, 2)),
            )
            faces.append(fi)

        frame_result = FrameResult(fi_idx, ts, faces)
        classifications = clf.classify_frame(faces)
        engine.add_frame(frame_result, classifications)

    print("[Demo] Session simulation complete. Computing metrics...")
    metrics = engine.finalize()

    _ensure_reports_dir()

    # Print metrics
    print(f"\n{'='*60}")
    print(f"  Demo Session Metrics")
    print(f"{'='*60}")
    print(f"  Class Attention Score:      {metrics.class_attention_score:.1f}/100")
    print(f"  Focus Stability Index:      {metrics.focus_stability_index:.3f}")
    print(f"  Disengagement Duration:     {metrics.disengagement_duration_pct:.1f}%")
    print(f"  Participation Equity Score: {metrics.participation_equity_score:.3f}")
    print(f"  Overall Level:              {metrics.engagement_level}")
    print(f"{'='*60}\n")

    dashboard_path = generate_dashboard(metrics, output_dir="reports", session_label="Demo Session")
    print(f"[Demo] Dashboard saved: {dashboard_path}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        generator = ReportGenerator(api_key=api_key)
        report = generator.generate(
            metrics,
            teacher_name="Demo Teacher",
            subject="Introduction to Thermodynamics",
        )
        report.save("reports/demo_session_report.md")
        report.print_summary()
    else:
        print("\n[Demo] Set ANTHROPIC_API_KEY to generate the AI narrative report.")
        print("       The dashboard chart has been saved to reports/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _print_banner()

    parser = argparse.ArgumentParser(
        description="AI Classroom Engagement Analyzer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["webcam", "video", "demo"],
        help=(
            "webcam  — Live webcam analysis\n"
            "video   — Analyze a recorded video file\n"
            "demo    — Generate a synthetic demo (no camera needed)"
        ),
    )
    parser.add_argument("video_path", nargs="?", help="Path to video file (required for 'video' mode)")
    parser.add_argument("--camera",     type=int,   default=0,    help="Webcam index (default: 0)")
    parser.add_argument("--session-id", type=str,   default=None, help="Session identifier")
    parser.add_argument("--teacher",    type=str,   default=None, help="Teacher name for report")
    parser.add_argument("--subject",    type=str,   default=None, help="Subject/topic taught")
    parser.add_argument("--sample",     type=int,   default=3,    help="Analyze every Nth frame (default: 3)")
    parser.add_argument("--no-preview", action="store_true",      help="Disable OpenCV preview window")

    args = parser.parse_args()

    show_preview = not args.no_preview

    if args.mode == "webcam":
        run_webcam_mode(
            camera_index=args.camera,
            session_id=args.session_id,
            teacher_name=args.teacher,
            subject=args.subject,
            show_preview=show_preview,
            sample_every=args.sample,
        )

    elif args.mode == "video":
        if not args.video_path:
            print("[Error] Please provide a video file path for 'video' mode.")
            print("  Example: python main.py video lecture.mp4")
            sys.exit(1)
        run_video_mode(
            video_path=args.video_path,
            session_id=args.session_id,
            teacher_name=args.teacher,
            subject=args.subject,
            show_preview=show_preview,
            sample_every=args.sample,
        )

    elif args.mode == "demo":
        run_demo_mode()


if __name__ == "__main__":
    main()
