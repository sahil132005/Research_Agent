# Research Brief Generator
### Multi-Step LLM Agent · Grok API · DuckDuckGo Search

---

## What the agent does

Given any research topic, the agent produces a structured, professional research brief
by running a 5-step pipeline where every step feeds its output into the next.

No single step can be removed without breaking the chain — the dependencies are explicit
and traceable in the code.

---

## Chain design (one step per function in `steps.py`)

| Step | Type | Input (from state) | Output (written to state) |
|------|------|--------------------|---------------------------|
| 1 | **LLM** — Topic Analysis | `topic` (raw user string) | `topic_analysis` dict: research angles, key terms, search queries |
| 2 | **Tool** — DuckDuckGo Search | `search_queries` (from Step 1) | `search_results` list of web snippets |
| 3 | **LLM** — Synthesis | `topic_analysis` + `search_results` | `synthesis` dict: findings per angle, overall summary |
| 4 | **LLM** — Critical Analysis | `topic_analysis` + `synthesis` | `critical_analysis` dict: insights, gaps, debates, confidence |
| 5 | **LLM** — Report Generation | All previous outputs | `final_report` (Markdown string) |

State at end of chain contains every intermediate artifact, saved as a JSON snapshot
alongside the final Markdown report.

---

## Installation

```bash
# 1. Clone or unzip the project
cd research_agent

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env_example .env
# Open .env and fill in one of:
#   GROK_API_KEY=your_grok_key_here
#   OPENAI_API_KEY=your_openai_key_here
```

Dependencies:
- `openai` — Grok's API is OpenAI-compatible; this is the only client library needed.
- `duckduckgo-search` — free, no API key required.
- `python-dotenv` — loads the .env file automatically at startup.

---

## How to run

```bash
# Option A — pass the topic as a command-line argument
python main.py "Quantum computing in cryptography"

# Option B — interactive prompt
python main.py
# Enter your research topic: Quantum computing in cryptography
```

Outputs are written to `output/`:
- `report_<topic>_<timestamp>.md` — the final research brief (open in any Markdown viewer)
- `state_<topic>_<timestamp>.json` — full pipeline state snapshot for inspection

---

## What inputs the agent expects

A single natural-language research topic string. Examples that work well:
- `"The role of gut microbiome in mental health"`
- `"Federated learning for medical imaging"`
- `"Carbon capture technologies 2024"`

The agent handles vague or broad topics but produces sharper reports for focused ones.

---

## File structure

```
research_agent/
├── main.py          # CLI entry point — run this
├── agent.py         # Pipeline orchestration (run_agent, save_outputs)
├── steps.py         # One function per pipeline step
├── llm.py           # Grok API wrapper + JSON response parser
├── tools.py         # DuckDuckGo web search tool
├── requirements.txt
├── README.md
└── output/          # Generated reports and state snapshots
```

---

## How to inspect what each step received and returned

Every step function in `steps.py` reads only from clearly named `state` keys and writes
to a clearly named output key. To trace Step 3 specifically:

```python
# In agent.py, add after step3_synthesize_findings(state):
import json
print(json.dumps(state["synthesis"], indent=2))
```

You can also open the `state_*.json` file after a run — it contains the output of every
step except the raw search results (which are omitted to keep the file readable).

---

## What happens if the tool (Step 2) fails

`step2_web_search` in `steps.py` wraps the DuckDuckGo call in a try/except.
- Per-query failures are caught and logged; remaining queries continue.
- If all queries fail, `state["search_results"]` is set to `[]` and `state["search_error"]`
  records the exception message.
- Step 3 detects the empty results list and instructs the LLM to synthesise from general
  knowledge, flagging this clearly in its output.
- The pipeline continues to completion — the final report will note the limitation.

---

## Known limitations

- DuckDuckGo rate-limits heavy usage; running many queries in quick succession may
  trigger temporary blocks.
- Snippet length varies; some sources return very short excerpts that reduce synthesis quality.
- Grok's context window caps how many search results Step 3 can process (currently capped at 14).
- The agent does not fetch full page content — only the search snippet.
