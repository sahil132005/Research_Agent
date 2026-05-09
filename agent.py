"""
agent.py — Pipeline orchestration.

run_agent(topic) initialises shared state, calls each step in sequence,
and returns the final state dictionary.

save_outputs(state) writes the Markdown report and a JSON state snapshot
to the output/ directory.

The state dictionary is the single source of truth for the entire pipeline.
Every step reads from it and writes back to it — no global variables,
no hidden side effects.
"""

import json
import os
from datetime import datetime

from steps import (
    step1_analyze_topic,
    step2_web_search,
    step3_synthesize_findings,
    step4_critical_analysis,
    step5_generate_report,
)


def run_agent(topic: str) -> dict:
    """
    Execute the full research pipeline for a given topic.

    Pipeline (5 steps):
      Step 1  — LLM   : Topic analysis → structured angles + search queries
      Step 2  — Tool  : DuckDuckGo web search → real-world snippets
      Step 3  — LLM   : Synthesise search results per research angle
      Step 4  — LLM   : Critical analysis → insights, gaps, controversies
      Step 5  — LLM   : Generate final Markdown research brief

    Args:
        topic: The user-supplied research topic string.

    Returns:
        The fully populated state dictionary.
    """
    _banner(f"Research Agent — topic: {topic!r}")

    # ── Shared state: every step reads from and writes to this dict ──────────
    state: dict = {
        "topic":            topic,
        "timestamp":        datetime.now().isoformat(),
        # Populated by steps:
        "topic_analysis":   None,   # Step 1
        "search_queries":   None,   # Step 1 → Step 2
        "search_results":   None,   # Step 2 (tool)
        "synthesis":        None,   # Step 3
        "critical_analysis":None,   # Step 4
        "final_report":     None,   # Step 5
    }
    # ────────────────────────────────────────────────────────────────────────

    try:
        state = step1_analyze_topic(state)
        state = step2_web_search(state)
        state = step3_synthesize_findings(state)
        state = step4_critical_analysis(state)
        state = step5_generate_report(state)
    except Exception as exc:
        state["error"] = str(exc)
        _banner("Pipeline aborted — see error above")
        raise

    _banner("Pipeline complete")
    return state


def save_outputs(state: dict, output_dir: str = "output") -> dict[str, str]:
    """
    Persist the final report (Markdown) and a state snapshot (JSON).

    Args:
        state:      The fully populated state dict returned by run_agent().
        output_dir: Directory to write files into (created if absent).

    Returns:
        Dict with keys 'report' and 'state' containing the file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = _safe_filename(state.get("topic", "unknown"), max_len=40)

    report_path = os.path.join(output_dir, f"report_{safe_topic}_{ts}.md")
    state_path  = os.path.join(output_dir, f"state_{safe_topic}_{ts}.json")

    # ── Markdown report ──────────────────────────────────────────────────────
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(state.get("final_report") or "# Error\n\nReport generation failed.")

    # ── JSON state snapshot (omit bulky raw search results) ─────────────────
    snapshot = {k: v for k, v in state.items() if k != "search_results"}
    snapshot["search_result_count"] = len(state.get("search_results") or [])

    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, ensure_ascii=False)

    return {"report": report_path, "state": state_path}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_filename(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text to a filesystem-safe filename fragment."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)
    return safe.strip().replace(" ", "_")[:max_len]


def _banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print(f"\n{'─' * width}")
    print(f"  {msg}")
    print(f"{'─' * width}")
