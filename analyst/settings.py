"""Configuration from environment variables, with a minimal .env loader."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

BACKEND = os.environ.get("AGENT_BACKEND", "anthropic")  # anthropic | openai
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))
# OpenAI-compatible servers, e.g. LM Studio (default) or llama.cpp / vLLM.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "lm-studio")
