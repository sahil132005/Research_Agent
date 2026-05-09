"""
steps.py — One function per pipeline step.

Each function:
  1. Reads from the shared `state` dictionary.
  2. Runs either an LLM call or an external tool.
  3. Writes its output back into `state`.
  4. Returns the updated `state`.

Data flow:
  step1 → search_queries ──────────────────────────────────────────┐
  step2 → search_results (tool output) ──────────────────────────┐ │
  step3 → synthesis  (uses step1 analysis + step2 results) ──┐   │ │
  step4 → critical_analysis (uses step1 + step3)           ──┤   │ │
  step5 → final_report (uses ALL previous outputs)           └───┴─┘
"""

import json
from llm import call_llm, parse_json_response
from tools import web_search as _web_search


# ---------------------------------------------------------------------------
# Step 1 — Topic Analysis (LLM)
# ---------------------------------------------------------------------------

_STEP1_SYSTEM = """You are a research analyst. Decompose a user-supplied research topic into
structured components that will drive a multi-step research pipeline.

Respond ONLY with a valid JSON object — no markdown fences, no explanation, no preamble.

Required structure (all fields mandatory):
{
  "topic_summary":     "<one-sentence description of the topic>",
  "research_angles":   ["<angle 1>", "<angle 2>", "<angle 3>"],
  "key_terms":         ["<term 1>", "<term 2>", "<term 3>", "<term 4>", "<term 5>"],
  "search_queries":    ["<query 1>", "<query 2>", "<query 3>", "<query 4>"],
  "complexity":        "<introductory | intermediate | advanced>",
  "domain":            "<academic / professional domain>"
}

Guidelines:
- research_angles : 3–4 distinct sub-questions or perspectives on the topic.
- key_terms       : 4–6 precise terms central to the topic.
- search_queries  : 3–5 targeted web-search strings, each under 10 words, covering
                    different angles so the search results are diverse.
- complexity      : honest assessment of the topic's technical depth."""


def step1_analyze_topic(state: dict) -> dict:
    """
    LLM Call 1 — Topic Analysis.

    Input  : state['topic'] (raw user string)
    Output : state['topic_analysis'] (structured dict)
             state['search_queries'] (list[str], forwarded to Step 2)
    """
    print("\n[Step 1 / 5] Analysing topic with LLM ...")

    user_prompt = f"Analyse this research topic and return the JSON structure:\n\n{state['topic']}"

    raw = call_llm(_STEP1_SYSTEM, user_prompt, temperature=0.3)

    try:
        analysis = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [Step 1 Warning] JSON parse failed ({exc}). Using minimal fallback.")
        analysis = {
            "topic_summary":  state["topic"],
            "research_angles": ["Overview", "Applications", "Challenges"],
            "key_terms":       [state["topic"]],
            "search_queries":  [state["topic"],
                                f"{state['topic']} overview",
                                f"{state['topic']} applications"],
            "complexity":      "intermediate",
            "domain":          "General",
        }

    state["topic_analysis"] = analysis
    state["search_queries"]  = analysis.get("search_queries", [state["topic"]])

    print(f"  → Domain     : {analysis.get('domain', '—')}")
    print(f"  → Angles     : {len(analysis.get('research_angles', []))}")
    print(f"  → Queries    : {len(state['search_queries'])}")
    return state


# ---------------------------------------------------------------------------
# Step 2 — Web Search (External Tool — DuckDuckGo)
# ---------------------------------------------------------------------------

def step2_web_search(state: dict) -> dict:
    """
    Tool Call — DuckDuckGo Web Search.

    Input  : state['search_queries'] (list[str] from Step 1)
    Output : state['search_results'] (list[dict] with title/url/snippet)

    Error handling: if search fails entirely, stores an empty list and a
    search_error key so subsequent steps can degrade gracefully.
    """
    print("\n[Step 2 / 5] Running web search tool ...")

    queries = state.get("search_queries", [state["topic"]])

    try:
        results = _web_search(queries, max_results_per_query=4)
        state["search_results"] = results
        print(f"  → Retrieved  : {len(results)} unique results")
    except Exception as exc:
        print(f"  [Tool Error] All searches failed: {exc}")
        state["search_results"] = []
        state["search_error"]   = str(exc)

    return state


# ---------------------------------------------------------------------------
# Step 3 — Synthesis (LLM)
# ---------------------------------------------------------------------------

_STEP3_SYSTEM = """You are a research synthesiser. Given a topic analysis and web-search
snippets, organise the findings by research angle.

Respond ONLY with a valid JSON object — no markdown fences, no explanation.

Required structure:
{
  "synthesis_per_angle": {
    "<angle name>": {
      "summary":            "<2–3 sentence synthesis for this angle>",
      "key_findings":       ["<finding 1>", "<finding 2>", "<finding 3>"],
      "sources_referenced": ["<url 1>", "<url 2>"]
    }
  },
  "overall_summary":  "<3–4 sentence paragraph covering all angles>",
  "notable_sources":  ["<url 1>", "<url 2>", "<url 3>"]
}

If search results are sparse or missing, synthesise from general knowledge but flag
this in each angle's summary with the phrase "(based on general knowledge)"."""


def step3_synthesize_findings(state: dict) -> dict:
    """
    LLM Call 2 — Synthesis.

    Input  : state['topic_analysis'] (Step 1)
             state['search_results'] (Step 2, tool output)
    Output : state['synthesis'] (structured dict)
    """
    print("\n[Step 3 / 5] Synthesising findings with LLM ...")

    analysis = state["topic_analysis"]
    results  = state["search_results"]

    if results:
        # Cap at 14 results to stay within reasonable context length
        results_block = "\n\n".join(
            f"[Result {i + 1}]\n"
            f"Query  : {r['query']}\n"
            f"Title  : {r['title']}\n"
            f"URL    : {r['url']}\n"
            f"Snippet: {r['snippet']}"
            for i, r in enumerate(results[:14])
        )
    else:
        results_block = (
            "No web results were retrieved. "
            "Synthesise using general knowledge and flag this clearly."
        )

    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Research Angles:\n{json.dumps(analysis['research_angles'], indent=2)}\n\n"
        f"Key Terms:\n{json.dumps(analysis['key_terms'], indent=2)}\n\n"
        f"Search Results:\n{results_block}\n\n"
        "Synthesise these results, organising findings by each research angle."
    )

    raw = call_llm(_STEP3_SYSTEM, user_prompt, temperature=0.5)

    try:
        synthesis = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [Step 3 Warning] JSON parse failed ({exc}). Storing raw text.")
        synthesis = {
            "overall_summary":     raw,
            "synthesis_per_angle": {},
            "notable_sources":     [],
        }

    state["synthesis"] = synthesis
    angle_count = len(synthesis.get("synthesis_per_angle", {}))
    print(f"  → Angles covered : {angle_count}")
    return state


# ---------------------------------------------------------------------------
# Step 4 — Critical Analysis (LLM)
# ---------------------------------------------------------------------------

_STEP4_SYSTEM = """You are a critical research analyst. Examine synthesised research findings
and produce a rigorous critical evaluation.

Respond ONLY with a valid JSON object — no markdown fences, no explanation.

Required structure:
{
  "key_insights":           ["<insight 1>", "<insight 2>", "<insight 3>"],
  "debates_controversies":  ["<debate 1>", "<debate 2>"],
  "knowledge_gaps":         ["<gap 1>", "<gap 2>"],
  "information_limitations":["<limitation 1>", "<limitation 2>"],
  "confidence_assessment":  "<low | medium | high>",
  "confidence_rationale":   "<1–2 sentences explaining the rating>"
}

Be specific and honest. Generic statements like 'more research is needed' do not qualify."""


def step4_critical_analysis(state: dict) -> dict:
    """
    LLM Call 3 — Critical Analysis.

    Input  : state['topic_analysis'] (Step 1)
             state['synthesis']      (Step 3)
    Output : state['critical_analysis'] (structured dict)
    """
    print("\n[Step 4 / 5] Performing critical analysis with LLM ...")

    user_prompt = (
        f"Original Topic: {state['topic']}\n\n"
        f"Topic Analysis:\n{json.dumps(state['topic_analysis'], indent=2)}\n\n"
        f"Synthesised Findings:\n{json.dumps(state['synthesis'], indent=2)}\n\n"
        "Perform a critical analysis of these findings."
    )

    raw = call_llm(_STEP4_SYSTEM, user_prompt, temperature=0.4)

    try:
        critical = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [Step 4 Warning] JSON parse failed ({exc}). Using fallback.")
        critical = {
            "key_insights":            [raw],
            "debates_controversies":   [],
            "knowledge_gaps":          [],
            "information_limitations": [],
            "confidence_assessment":   "medium",
            "confidence_rationale":    "Parse error — see raw text in key_insights.",
        }

    state["critical_analysis"] = critical
    print(f"  → Key insights : {len(critical.get('key_insights', []))}")
    print(f"  → Confidence   : {critical.get('confidence_assessment', '—')}")
    return state


# ---------------------------------------------------------------------------
# Step 5 — Report Generation (LLM)
# ---------------------------------------------------------------------------

_STEP5_SYSTEM = """You are a professional research report writer. Compile a polished,
well-structured research brief in Markdown.

Use proper Markdown: ATX headers (##), bullet lists, bold for emphasis.
Be precise and avoid filler phrases.

Include EXACTLY these sections in this order:
1. ## Executive Summary
2. ## Topic Overview
3. ## Research Findings  (one sub-section ### per research angle)
4. ## Key Insights
5. ## Ongoing Debates & Controversies
6. ## Knowledge Gaps
7. ## Limitations of This Research
8. ## Conclusion

Output ONLY the Markdown — no preamble, no code fences."""


def step5_generate_report(state: dict) -> dict:
    """
    LLM Call 4 — Final Report Generation.

    Input  : ALL previous step outputs (topic_analysis, synthesis, critical_analysis)
    Output : state['final_report'] (Markdown string, also written to file by agent.py)
    """
    print("\n[Step 5 / 5] Generating final Markdown report with LLM ...")

    user_prompt = (
        f"TOPIC: {state['topic']}\n\n"
        f"TOPIC ANALYSIS:\n{json.dumps(state['topic_analysis'], indent=2)}\n\n"
        f"SYNTHESISED FINDINGS:\n{json.dumps(state['synthesis'], indent=2)}\n\n"
        f"CRITICAL ANALYSIS:\n{json.dumps(state['critical_analysis'], indent=2)}\n\n"
        "Write the full research brief in Markdown now."
    )

    report_md = call_llm(_STEP5_SYSTEM, user_prompt, temperature=0.6)

    # Append a Sources section using URLs collected during synthesis
    notable = state["synthesis"].get("notable_sources", [])
    if notable:
        sources_md = "\n\n## Sources\n" + "\n".join(f"- <{u}>" for u in notable)
        report_md += sources_md

    state["final_report"] = report_md
    print(f"  → Report length: {len(report_md):,} characters")
    return state
