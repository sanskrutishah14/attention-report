"""
report_generator.py
--------------------
Uses the Anthropic Claude API to convert session metrics into a structured,
actionable teaching report in natural language.

The report includes:
  - Session overview (engagement level, KPIs)
  - Attention timeline narrative
  - Low engagement period analysis
  - Participation equity insights
  - 5 concrete, prioritized teaching recommendations
  - Optional: week-over-week trend if previous sessions provided

Usage:
    generator = ReportGenerator(api_key="sk-ant-...")
    report = generator.generate(session_metrics)
    print(report.full_text)
    report.save("reports/session_01.md")
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import anthropic

from metrics_engine import SessionMetrics


# ---------------------------------------------------------------------------
# Report Data Structure
# ---------------------------------------------------------------------------

@dataclass
class SessionReport:
    session_id: str
    generated_at: str
    engagement_level: str
    class_attention_score: float
    full_text: str                         # Full markdown report from Claude
    key_recommendations: List[str]         # Extracted bullet recommendations
    raw_metrics_json: str                  # Embedded metrics for archiving

    def save(self, path: str):
        """Save the markdown report to a file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.full_text)
        print(f"[ReportGenerator] Saved report to: {path}")

    def print_summary(self):
        """Print a concise terminal summary using rich if available."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box

            console = Console()
            console.print()
            console.print(Panel.fit(
                f"[bold cyan]Session: {self.session_id}[/bold cyan]\n"
                f"Engagement Level: [bold {'green' if self.engagement_level == 'High' else 'yellow' if self.engagement_level == 'Moderate' else 'red'}]{self.engagement_level}[/bold {'green' if self.engagement_level == 'High' else 'yellow' if self.engagement_level == 'Moderate' else 'red'}]\n"
                f"Class Attention Score: [bold]{self.class_attention_score:.1f}/100[/bold]",
                title="[bold]Classroom Engagement Report[/bold]",
                border_style="cyan"
            ))

            if self.key_recommendations:
                table = Table(title="Top Recommendations", box=box.ROUNDED, border_style="cyan")
                table.add_column("#", style="dim", width=3)
                table.add_column("Recommendation", style="white")
                for i, rec in enumerate(self.key_recommendations, 1):
                    table.add_row(str(i), rec)
                console.print(table)

            console.print()
        except ImportError:
            print(f"\n=== Session: {self.session_id} ===")
            print(f"Engagement Level: {self.engagement_level}")
            print(f"Class Attention Score: {self.class_attention_score:.1f}/100")
            print("\nTop Recommendations:")
            for i, rec in enumerate(self.key_recommendations, 1):
                print(f"  {i}. {rec}")
            print()


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates natural language session reports via Claude API.

    Args:
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        model: Claude model to use.
        max_tokens: Max tokens for the report.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        max_tokens: int = 2000,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or pass api_key= to ReportGenerator()."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model  = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        metrics: SessionMetrics,
        teacher_name: Optional[str] = None,
        subject: Optional[str] = None,
        previous_sessions: Optional[List[SessionMetrics]] = None,
    ) -> SessionReport:
        """
        Generate a full session report.

        Args:
            metrics: SessionMetrics from MetricsEngine.finalize()
            teacher_name: Optional teacher name for personalization.
            subject: Optional subject/topic taught.
            previous_sessions: Optional list of previous SessionMetrics for trend analysis.

        Returns:
            SessionReport with full markdown text and extracted recommendations.
        """
        prompt = self._build_prompt(metrics, teacher_name, subject, previous_sessions)
        system = self._system_prompt()

        print("[ReportGenerator] Calling Claude API to generate report...")
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        full_text = message.content[0].text
        recommendations = self._extract_recommendations(full_text)

        return SessionReport(
            session_id=metrics.session_id,
            generated_at=datetime.now().isoformat(),
            engagement_level=metrics.engagement_level,
            class_attention_score=metrics.class_attention_score,
            full_text=full_text,
            key_recommendations=recommendations,
            raw_metrics_json=json.dumps(self._metrics_dict(metrics), indent=2),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        return """You are an expert educational consultant and classroom engagement analyst.
Your role is to help teachers improve their classroom effectiveness by interpreting
engagement data collected from anonymous, computer-vision-based monitoring.

Guidelines for your reports:
- Be warm, constructive, and actionable — never critical or judgmental of teachers or students.
- Focus on patterns, not individuals. All analysis is at the class level.
- Ground recommendations in evidence-based teaching strategies (active recall,
  spaced repetition, pacing changes, interactive activities).
- Use clear section headers and bullet points for readability.
- Keep the tone professional but conversational.
- Never suggest surveillance or punitive use of this data.
- Acknowledge the limitations of automated engagement detection.

Format: Respond in well-structured Markdown with clear sections."""

    def _build_prompt(
        self,
        m: SessionMetrics,
        teacher_name: Optional[str],
        subject: Optional[str],
        previous: Optional[List[SessionMetrics]],
    ) -> str:

        teacher_str = f"Teacher: {teacher_name}" if teacher_name else "Teacher: (not specified)"
        subject_str = f"Subject: {subject}" if subject else "Subject: (not specified)"
        duration_min = m.duration_sec / 60

        # Timeline as readable string
        timeline_str = ""
        if m.attention_timeline:
            timeline_str = "\n".join(
                f"  Minute {t['minute']:2d}: {t['avg_score']:5.1f}/100  [{t['dominant_state']}]"
                for t in m.attention_timeline
            )
        else:
            timeline_str = "  (no timeline data)"

        # State distribution
        dist_str = "\n".join(
            f"  {state}: {pct:.1f}%"
            for state, pct in m.state_distribution.items()
        )

        # Peak distraction windows
        if m.peak_distraction_windows:
            windows_str = "\n".join(
                f"  {self._sec_to_time(w['start_sec'])}–{self._sec_to_time(w['end_sec'])} "
                f"({w['duration_sec']:.0f}s, avg score {w['avg_score']:.0f}/100)"
                for w in m.peak_distraction_windows
            )
        else:
            windows_str = "  No significant low-engagement windows detected."

        # Previous session trend
        trend_str = ""
        if previous:
            trend_str = "\n\nPREVIOUS SESSION COMPARISON:\n"
            for prev in previous[-3:]:   # last 3 sessions
                trend_str += (
                    f"  Session {prev.session_id}: "
                    f"CAS={prev.class_attention_score:.1f}, "
                    f"FSI={prev.focus_stability_index:.2f}, "
                    f"PES={prev.participation_equity_score:.2f}\n"
                )

        prompt = f"""Please generate a comprehensive classroom engagement report based on the
anonymized engagement data collected during this lecture session.

SESSION DETAILS:
  {teacher_str}
  {subject_str}
  Session ID: {m.session_id}
  Duration: {duration_min:.1f} minutes
  Avg students detected per frame: {m.avg_students_detected:.0f}
  Total frames analyzed: {m.total_frames_analyzed}

KEY PERFORMANCE METRICS:
  Class Attention Score (CAS): {m.class_attention_score:.1f}/100
  → Average engagement level across the full session

  Focus Stability Index (FSI): {m.focus_stability_index:.3f}  (scale 0–1, higher = more consistent)
  → Measures how stable engagement was (vs. high variance/fluctuation)

  Disengagement Duration: {m.disengagement_duration_pct:.1f}% of session
  → % of time where class average engagement fell below 40%

  Participation Equity Score (PES): {m.participation_equity_score:.3f}  (scale 0–1, higher = more equitable)
  → Reflects how evenly distributed attention was across the class

  Overall Engagement Level: {m.engagement_level}

ATTENTION STATE DISTRIBUTION:
{dist_str}

ATTENTION TIMELINE (per minute):
{timeline_str}

PEAK LOW-ENGAGEMENT WINDOWS:
{windows_str}{trend_str}

---

Please write a full report with the following sections:

## Session Overview
Summarise the overall engagement level, what the KPIs suggest about the session,
and one headline observation.

## Attention Patterns
Discuss the attention timeline, when engagement was highest and lowest, and what
patterns are visible (e.g., mid-session dip, strong start but tired ending).

## Low Engagement Analysis
If there were significant disengagement windows, discuss what typically causes these
patterns and what they might indicate (content difficulty, pacing, fatigue, etc.).

## Participation Equity
Interpret the PES score. Was engagement spread relatively evenly, or were there
signs of uneven participation? What are the teaching implications?

## Recommendations
Provide exactly 5 specific, actionable recommendations for the teacher to improve
engagement in future sessions. Each recommendation should:
- Reference the specific metric or pattern that motivated it
- Name a concrete teaching technique or strategy
- Be implementable without technology

Format each recommendation as a numbered list item starting with a bold action verb.

## Limitations Note
Briefly note the limitations of automated engagement detection (1–2 sentences).
"""
        return prompt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sec_to_time(seconds: float) -> str:
        """Convert seconds to MM:SS string."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _extract_recommendations(text: str) -> List[str]:
        """
        Extract numbered recommendations from the markdown report.
        Returns up to 5 clean recommendation strings.
        """
        recs = []
        # Look for numbered list items in the Recommendations section
        pattern = re.compile(r'^\s*\d+\.\s+(.+)', re.MULTILINE)

        # Try to isolate the Recommendations section first
        rec_section_match = re.search(
            r'##\s+Recommendations.*?(?=##|\Z)', text, re.DOTALL | re.IGNORECASE
        )
        search_text = rec_section_match.group(0) if rec_section_match else text

        for m in pattern.finditer(search_text):
            line = m.group(1).strip()
            # Strip markdown bold markers
            line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
            if len(line) > 10:
                recs.append(line)
            if len(recs) >= 5:
                break

        return recs

    @staticmethod
    def _metrics_dict(m: SessionMetrics) -> dict:
        return {
            "session_id": m.session_id,
            "duration_sec": m.duration_sec,
            "class_attention_score": m.class_attention_score,
            "focus_stability_index": m.focus_stability_index,
            "disengagement_duration_pct": m.disengagement_duration_pct,
            "participation_equity_score": m.participation_equity_score,
            "engagement_level": m.engagement_level,
            "avg_students_detected": m.avg_students_detected,
            "state_distribution": m.state_distribution,
        }
