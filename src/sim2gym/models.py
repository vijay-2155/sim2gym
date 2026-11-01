"""Model clients.

- ``openrouter:<model-id>`` (MAIN backend; free models): OpenAI client against https://openrouter.ai/api/v1,
  key from env ``OPENROUTER_API_KEY`` (or the gitignored repo-root ``.env``). 429s back off honouring
  Retry-After (at most 5 retries), then the call fails with ``error="rate_limited"``.
- ``bedrock:<modelId>`` (paid; short tests only, see runner's paid guard): boto3 ``converse`` in
  ``SIM2GYM_BEDROCK_REGION`` (default us-east-1) with the local AWS CLI credential chain.
- ``openai:<model>``: any OpenAI-compatible server, ``SIM2GYM_BASE_URL`` (default Ollama
  http://localhost:11434/v1) and ``SIM2GYM_API_KEY``.

Keys and endpoints come only from the server-side environment, never from HTTP clients, and are never
logged, returned or saved (``settings()`` omits them).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MAX_TOKENS = 2048
REASONING_MAX_TOKENS = 6144  # gpt-oss/magistral/deepseek/kimi spend tokens on reasoning before the text block
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRIES = 4

# Server-side allowlist (API + live demo dropdown): Bedrock ids verified working in us-east-1, 2026-11-01.
BEDROCK_MODELS = [
    "qwen.qwen3-32b-v1:0",
    "qwen.qwen3-next-80b-a3b",
    "qwen.qwen3-coder-30b-a3b-v1:0",
    "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-20b-1:0",
    "mistral.mistral-large-3-675b-instruct",
    "mistral.magistral-small-2509",
    "nvidia.nemotron-super-3-120b",
    "deepseek.v3.2",
    "moonshotai.kimi-k2.5",
    "google.gemma-3-27b-it",
    "google.gemma-3-12b-it",
    "google.gemma-3-4b-it",
]
FALLBACK_DEMO_MODEL = "6-probe"  # used when no free OpenRouter model list is available
REASONING_PREFIXES = ("openai.gpt-oss", "mistral.magistral", "deepseek.", "moonshotai.kimi")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MAX_RETRIES = 5
OPENROUTER_MAX_WAIT_S = 60.0  # a longer Retry-After (e.g. daily free cap) fails fast instead of waiting
OPENROUTER_MAX_TOTAL_WAIT_S = 60.0  # total backoff sleep per call
OPENROUTER_MAX_TOKENS = 12000  # free Nemotron models spend ~3-4k tokens on `reasoning`; 4096 gave empty content
DEMO_FREE_MODEL = "nvidia/nemotron-3.5-super-vl:free"  # verified working 2026-11-01, ~9-25 s/turn
# Not usable for this benchmark: harness-only (403), music, safety classifiers, the auto router.
FREE_EXCLUDE = (re.compile(r"^thinkingmachines/inkling"), re.compile(r"^google/lyria-"),
                re.compile(r"content-safety"), re.compile(r"^openrouter/free$"))
FREE_CACHE_TTL_S = 600.0
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Tiny .env reader: KEY=VALUE lines, optional quotes, never overrides the real environment."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k and v:
            os.environ.setdefault(k, v)


load_dotenv()


def openrouter_base_url() -> str:
    return os.environ.get("SIM2GYM_OPENROUTER_BASE_URL", OPENROUTER_URL).rstrip("/")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)


@dataclass
class ChatResult:
    text: str
    latency_s: float
    reasoning: str | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    detail: str | None = None  # extra context for error (e.g. upstream message); never contains keys


def split_think(text: str) -> tuple[str, str | None]:
    """Remove <think>...</think> blocks (qwen3 / deepseek on Ollama) from the protocol text."""
    if not text:
        return "", None
    thoughts = _THINK_RE.findall(text)
    if not thoughts:
        return text, None
    return _THINK_RE.sub("", text).strip(), "\n".join(t.strip() for t in thoughts)


def is_reasoning_model(model_id: str) -> bool:
    return model_id.startswith(REASONING_PREFIXES)


class BedrockModel:
    kind = "bedrock"

    def __init__(self, model_id: str, max_tokens: int | None = None, temperature: float = 0.0,
                 timeout_s: float = DEFAULT_TIMEOUT_S, retries: int = DEFAULT_RETRIES, region: str | None = None,
                 client: Any = None):
        self.model_id = model_id
        self.max_tokens = int(max_tokens or (REASONING_MAX_TOKENS if is_reasoning_model(model_id) else DEFAULT_MAX_TOKENS))
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.region = region or os.environ.get("SIM2GYM_BEDROCK_REGION", "us-east-1")
        self._client = client

    @property
    def name(self) -> str:
        return f"bedrock:{self.model_id}"

    def settings(self) -> dict:
        return {"provider": "bedrock", "model_id": self.model_id, "region": self.region,
                "temperature": self.temperature, "max_tokens": self.max_tokens, "timeout_s": self.timeout_s}

    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config
            cfg = Config(read_timeout=self.timeout_s, connect_timeout=15,
                         retries={"max_attempts": 2, "mode": "standard"})
            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)
        return self._client

    def chat(self, messages: list[dict]) -> ChatResult:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        conv = [{"role": m["role"], "content": [{"text": m["content"] or "(empty)"}]}
                for m in messages if m["role"] in ("user", "assistant")]
        kwargs: dict[str, Any] = {"modelId": self.model_id, "messages": conv,
                                  "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": self.temperature}}
        if system:
            kwargs["system"] = system
        t0 = time.monotonic()
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.client().converse(**kwargs)
                break
            except Exception as exc:  # botocore ClientError / timeouts
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
                retryable = code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                                     "InternalServerException", "TooManyRequestsException") \
                    or type(exc).__name__ in ("ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError")
                last_err = f"{type(exc).__name__}: {code or exc}"
                if not retryable or attempt == self.retries:
                    return ChatResult("", time.monotonic() - t0, error=f"bedrock error: {last_err}"[:500])
                time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        latency = time.monotonic() - t0
        blocks = resp.get("output", {}).get("message", {}).get("content", []) or []
        texts, reasoning = [], []
        for b in blocks:
            if "text" in b:
                texts.append(b["text"])
            elif "reasoningContent" in b:
                rt = b["reasoningContent"].get("reasoningText", {}).get("text")
                if rt:
                    reasoning.append(rt)
        text, think = split_think("\n".join(texts))
        if think:
            reasoning.append(think)
        return ChatResult(text=text, latency_s=latency, reasoning="\n".join(reasoning) or None,
                          usage=dict(resp.get("usage", {})), stop_reason=resp.get("stopReason"))


class OpenAICompatModel:
    """OpenAI-compatible chat client with its own retry loop (429 honours Retry-After)."""
    kind = "openai"
    prefix = "openai"
    max_wait_s = OPENROUTER_MAX_WAIT_S
    max_total_wait_s = OPENROUTER_MAX_TOTAL_WAIT_S
    default_max_tokens = DEFAULT_MAX_TOKENS

    def __init__(self, model: str, max_tokens: int | None = None, temperature: float = 0.0,
                 timeout_s: float = DEFAULT_TIMEOUT_S, retries: int = 2, base_url: str | None = None,
                 api_key: str | None = None):
        self.model = model
        self.max_tokens = int(max_tokens or max(self.default_max_tokens,
                                                REASONING_MAX_TOKENS if _reasoning_any(model) else 0))
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.base_url = base_url or os.environ.get("SIM2GYM_BASE_URL", "http://localhost:11434/v1")
        self._api_key = api_key or os.environ.get("SIM2GYM_API_KEY", "ollama")
        self._client = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.prefix}:{self.model}"

    def settings(self) -> dict:  # no key here: settings are saved into results
        return {"provider": self.kind, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "max_tokens": self.max_tokens, "timeout_s": self.timeout_s}

    def _headers(self) -> dict:
        return {}

    def client(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(base_url=self.base_url, api_key=self._api_key, timeout=self.timeout_s,
                                      max_retries=0, default_headers=self._headers() or None)
            return self._client

    @staticmethod
    def _retry_after(exc) -> float | None:
        resp = getattr(exc, "response", None)
        h = getattr(resp, "headers", None) or {}
        v = h.get("retry-after") or h.get("Retry-After")
        try:
            return float(v) if v is not None else None
        except ValueError:
            return None

    def chat(self, messages: list[dict]) -> ChatResult:
        """429 -> retry honouring Retry-After, else ``rate_limited``; 5xx / timeouts / error bodies
        (choices=None) -> retry with backoff, else ``provider_error``. At most ``retries`` retries and
        ~``max_total_wait_s`` of sleeping in total, so a throttled free tier is not hammered."""
        import openai
        t0 = time.monotonic()
        slept = 0.0
        for attempt in range(self.retries + 1):
            kind, detail, wait = None, "", None
            try:
                resp = self.client().chat.completions.create(model=self.model, messages=messages,
                                                             temperature=self.temperature, max_tokens=self.max_tokens)
                choices = getattr(resp, "choices", None)
                if not choices or getattr(choices[0], "message", None) is None:
                    err = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
                    kind, detail = "provider_error", f"no choices in response: {str(err)[:200]}"
                else:
                    return self._result(resp, time.monotonic() - t0)
            except openai.RateLimitError as exc:
                kind, detail, wait = "rate_limited", str(exc)[:200], self._retry_after(exc)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                kind, detail = "provider_error", f"{type(exc).__name__}: {exc}"[:200]
            except openai.APIStatusError as exc:
                if getattr(exc, "status_code", 0) in (502, 503, 504, 529):
                    kind, detail = "provider_error", f"{type(exc).__name__}: {exc}"[:200]
                else:  # 4xx (bad model id, 403 harness-only, ...): not retryable
                    return ChatResult("", time.monotonic() - t0, error=f"{self.kind} error: {type(exc).__name__}: {exc}"[:500])
            except Exception as exc:
                return ChatResult("", time.monotonic() - t0, error=f"{self.kind} error: {type(exc).__name__}: {exc}"[:500])
            wait = min(30.0, 2.0 * 2 ** attempt + random.random()) if wait is None else wait
            if attempt == self.retries or wait > self.max_wait_s or slept + wait > self.max_total_wait_s:
                return ChatResult("", time.monotonic() - t0, error=kind, detail=detail)
            time.sleep(wait)
            slept += wait
        return ChatResult("", time.monotonic() - t0, error="provider_error")

    def _result(self, resp, latency: float) -> ChatResult:
        choice = resp.choices[0]
        raw = choice.message.content or ""  # only `content` is parsed; reasoning is recorded, never fed back
        extra = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
        if extra is None:
            extra = (getattr(choice.message, "model_extra", None) or {}).get("reasoning")
        text, think = split_think(raw)
        reasoning = "\n".join(x for x in (extra, think) if isinstance(x, str) and x) or None
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else {}
        return ChatResult(text=text, latency_s=latency, reasoning=reasoning, usage=usage,
                          stop_reason=getattr(choice, "finish_reason", None))


def _reasoning_any(model: str) -> bool:
    m = model.lower()
    return any(x in m for x in ("gpt-oss", "magistral", "deepseek", "kimi", "-r1", "qwq", "thinking"))


class OpenRouterModel(OpenAICompatModel):
    kind = "openrouter"
    prefix = "openrouter"
    default_max_tokens = OPENROUTER_MAX_TOKENS
    timeout_default = 180.0

    def __init__(self, model: str, retries: int = OPENROUTER_MAX_RETRIES, **kw):
        kw.pop("base_url", None)
        kw.pop("api_key", None)  # never from callers: env / .env only
        kw.setdefault("timeout_s", self.timeout_default)
        super().__init__(model, retries=retries, base_url=openrouter_base_url(),
                         api_key=os.environ.get("OPENROUTER_API_KEY") or "missing-key", **kw)
        if not os.environ.get("OPENROUTER_API_KEY"):
            self._missing_key = True

    def _headers(self) -> dict:
        return {"HTTP-Referer": "https://github.com/sim2gym", "X-Title": "sim2gym"}

    def chat(self, messages):
        if getattr(self, "_missing_key", False):
            return ChatResult("", 0.0, error="openrouter error: OPENROUTER_API_KEY not set (env or .env)")
        return super().chat(messages)


# ------------------------------------------------------------------ OpenRouter discovery / quota
_free_cache: dict = {"t": 0.0, "models": None, "error": None}
_free_lock = threading.Lock()
PREFERRED_FREE = ("qwen", "deepseek", "llama", "gpt-oss")


def _get_json(url: str, headers: dict | None = None, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "sim2gym", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _is_free(m: dict) -> bool:
    p = m.get("pricing") or {}
    return str(m.get("id", "")).endswith(":free") or (str(p.get("prompt")) == "0" and str(p.get("completion")) == "0")


def openrouter_free_models(force: bool = False) -> list[dict]:
    """Free OpenRouter models ({id, name, context_length}), cached 10 min. Empty list if unreachable."""
    with _free_lock:
        if not force and _free_cache["models"] is not None and time.monotonic() - _free_cache["t"] < FREE_CACHE_TTL_S:
            return list(_free_cache["models"])
        try:
            data = _get_json(openrouter_base_url() + "/models").get("data", [])
            models = [{"id": m["id"], "name": m.get("name") or m["id"], "context_length": m.get("context_length")}
                      for m in data if isinstance(m, dict) and m.get("id") and _is_free(m)
                      and not any(rx.search(m["id"]) for rx in FREE_EXCLUDE)]
            models.sort(key=lambda m: m["id"])
            _free_cache.update(t=time.monotonic(), models=models, error=None)
        except Exception as exc:
            _free_cache.update(t=time.monotonic(), models=[], error=f"{type(exc).__name__}: {exc}"[:200])
        return list(_free_cache["models"])


def free_models_error() -> str | None:
    return _free_cache["error"]


def default_free_model(free: list[dict]) -> str | None:
    if any(m["id"] == DEMO_FREE_MODEL for m in free):
        return "openrouter:" + DEMO_FREE_MODEL
    for fam in PREFERRED_FREE:
        for m in free:
            if fam in m["id"].lower():
                return "openrouter:" + m["id"]
    return ("openrouter:" + free[0]["id"]) if free else None


QUOTA_FIELDS = ("limit", "limit_remaining", "limit_reset", "usage", "usage_daily", "usage_weekly", "usage_monthly",
                "is_free_tier", "rate_limit")


def openrouter_quota() -> dict:
    """GET /key; returns only whitelisted limit/usage fields (never the key or its label)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {"error": "OPENROUTER_API_KEY not set (env or .env)"}
    try:
        data = _get_json(openrouter_base_url() + "/key", headers={"Authorization": f"Bearer {key}"}).get("data", {})
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {getattr(exc, 'code', '') or exc}"[:200]}
    return {k: data[k] for k in QUOTA_FIELDS if k in data}


def get_model(spec: str, **settings):
    """``openrouter:<id>``, ``bedrock:<modelId>`` or ``openai:<model>``. Other prefixes are rejected."""
    for prefix, cls in (("openrouter:", OpenRouterModel), ("bedrock:", BedrockModel), ("openai:", OpenAICompatModel)):
        if spec.startswith(prefix) and len(spec) > len(prefix):
            return cls(spec[len(prefix):], **settings)
    raise ValueError(f"unknown model spec {spec!r}; use openrouter:<id>, bedrock:<modelId>, openai:<model> "
                     "or a baseline name")
