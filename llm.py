"""
llm.py — LLM client supporting both Grok and OpenAI APIs.

Priority order for key detection:
  1. GROK_API_KEY   → uses https://api.x.ai/v1  (default model: grok-3)
  2. OPENAI_API_KEY → uses https://api.openai.com/v1 (default model: gpt-4o)

An optional MODEL variable in .env overrides the default model for whichever
provider is active.

All LLM calls in the pipeline route through call_llm().
parse_json_response() is a shared utility for steps that expect structured JSON.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load .env once at import time so every module benefits automatically
load_dotenv()

# ── Provider configuration ────────────────────────────────────────────────────

_PROVIDERS = {
    "grok": {
        "env_key":      "GROK_API_KEY",
        "base_url":     "https://api.x.ai/v1",
        "default_model":"grok-3",
        "label":        "Grok (xAI)",
    },
    "openai": {
        "env_key":      "OPENAI_API_KEY",
        "base_url":     None,               # OpenAI SDK uses its own default
        "default_model":"gpt-4o",
        "label":        "OpenAI",
    },
}


def _resolve_provider() -> tuple[str, str, str | None, str]:
    """
    Auto-detect which provider to use based on available environment variables.

    Returns:
        (api_key, model, base_url, label)

    Raises:
        EnvironmentError if neither key is set.
    """
    # Allow an explicit MODEL override from .env
    model_override = os.environ.get("MODEL", "").strip() or None

    for name, cfg in _PROVIDERS.items():
        api_key = os.environ.get(cfg["env_key"], "").strip()
        if api_key:
            model    = model_override or cfg["default_model"]
            base_url = cfg["base_url"]
            label    = cfg["label"]
            return api_key, model, base_url, label

    raise EnvironmentError(
        "\n\nNo API key found. Please open .env and fill in one of:\n"
        "  GROK_API_KEY=<your-grok-key>      (from https://console.x.ai)\n"
        "  OPENAI_API_KEY=<your-openai-key>  (from https://platform.openai.com)\n"
    )


def _get_client() -> tuple[OpenAI, str]:
    """
    Build and return an OpenAI-compatible client plus the resolved model name.

    Both Grok and OpenAI use the same openai SDK — only the base_url differs.
    """
    api_key, model, base_url, label = _resolve_provider()

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    return client, model, label


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """
    Send a chat completion request to whichever provider is configured.

    Args:
        system_prompt: Instruction that frames the model's role and output format.
        user_prompt:   The actual content/question for this step.
        model:         Override the resolved model for this specific call (optional).
        temperature:   Sampling temperature.

    Returns:
        The raw text content of the model's response.
    """
    client, resolved_model, label = _get_client()
    active_model = model or resolved_model

    response = client.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def print_active_provider() -> None:
    """Print which provider and model will be used — called once at startup."""
    try:
        _, model, label = _get_client()
        print(f"  Provider : {label}")
        print(f"  Model    : {model}")
    except EnvironmentError as exc:
        print(str(exc))
        raise


# ── JSON response parser ──────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    """
    Safely extract a JSON object from an LLM response.

    Handles three common formats:
      1. Pure JSON            → {"key": "value"}
      2. Markdown JSON block  → ```json\\n{"key": "value"}\\n```
      3. Markdown plain block → ```\\n{"key": "value"}\\n```

    Raises json.JSONDecodeError if none of the formats parse cleanly.
    """
    text = raw.strip()

    if "```" in text:
        for segment in text.split("```"):
            segment = segment.strip()
            if segment.lower().startswith("json"):
                segment = segment[4:].strip()
            if segment.startswith("{") or segment.startswith("["):
                try:
                    return json.loads(segment)
                except json.JSONDecodeError:
                    continue

    return json.loads(text)
