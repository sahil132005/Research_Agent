"""
main.py — Command-line entry point for the Research Brief Generator.

Usage:
    python main.py "your research topic here"
    python main.py                              # prompts interactively

The script runs the full 5-step pipeline and writes outputs to output/.
"""

import sys
from agent import run_agent, save_outputs
from llm import print_active_provider


def main() -> None:
    _print_header()
    print_active_provider()
    print()

    # ── Accept topic from CLI arg or interactive prompt ──────────────────────
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:]).strip()
        print(f"Topic (from args): {topic!r}\n")
    else:
        topic = input("Enter your research topic: ").strip()

    if not topic:
        print("Error: topic cannot be empty.")
        sys.exit(1)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    state = run_agent(topic)

    # ── Save outputs ─────────────────────────────────────────────────────────
    paths = save_outputs(state)

    print(f"\n✓  Report  → {paths['report']}")
    print(f"✓  State   → {paths['state']}\n")

    # ── Print a short preview of the report to the terminal ──────────────────
    report = state.get("final_report", "No report generated.")
    preview_limit = 1_200
    print("─" * 60)
    print("REPORT PREVIEW")
    print("─" * 60)
    if len(report) > preview_limit:
        print(report[:preview_limit])
        print(f"\n... [truncated — full report in {paths['report']}]")
    else:
        print(report)


def _print_header() -> None:
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║       Research Brief Generator               ║")
    print("║  Multi-Step LLM Agent  ·  Grok API           ║")
    print("╚══════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
