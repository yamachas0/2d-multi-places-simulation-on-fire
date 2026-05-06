"""
Provider-switching factory for LLM clients.

Each client exposes the same duck-typed surface as the original ClaudeClient:
  - generate(system_prompt, user_prompt, temperature=None, max_tokens=None) -> str
  - check_connection() -> bool
  - check_model_exists() -> bool
  - attributes: model, base_url

Supported providers: "anthropic" (default), "openai", "google".
"""
import datetime
import hashlib
import os
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RATE_LIMIT_RETRIES = 4


def _retry_backoff(attempt: int) -> int:
    return 2 ** attempt  # 1, 2, 4, 8


# Gemini structured-output schemas. Auto-selected by scanning system prompt for
# the "action_type" marker (present in the decision/action prompt only).
_GEMINI_SCHEMA_MESSAGE = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["message", "reasoning"],
}

# Used when skip_decision_prompt is on: message JSON also carries memory
# (since Phase 3 is stubbed and memory must be captured here).
_GEMINI_SCHEMA_MESSAGE_WITH_MEMORY = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "reasoning": {"type": "string"},
        "memory": {"type": "string"},
    },
    "required": ["message", "reasoning", "memory"],
}

_GEMINI_SCHEMA_ACTION = {
    "type": "object",
    "properties": {
        "action_type": {"type": "string"},
        "direction": {"type": "string"},
        "target_place": {"type": "string"},
        "target_agent": {"type": "string"},
        "memory": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["action_type", "memory", "reasoning"],
}


def _pick_gemini_schema(system_prompt: str) -> dict:
    if "action_type" in system_prompt:
        return _GEMINI_SCHEMA_ACTION
    # Heuristic: skip_decision-mode message prompt asks for "memory" inline.
    if '"memory"' in system_prompt:
        return _GEMINI_SCHEMA_MESSAGE_WITH_MEMORY
    return _GEMINI_SCHEMA_MESSAGE


class OpenAIClient:
    """OpenAI API client. Uses Chat Completions with system/user split."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "gpt-5.4-nano",
        temperature: float = 0.7,
        max_tokens: int = 200,
    ):
        from openai import OpenAI
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set")
            self._client = None
        else:
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._client is None:
            return ""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        last_err = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    logger.info(
                        "Token usage (openai): input=%s, cache_read=%s, output=%s",
                        getattr(usage, "prompt_tokens", 0),
                        getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0),
                    )
                choice = resp.choices[0] if resp.choices else None
                if choice and choice.message and choice.message.content:
                    return choice.message.content.strip()
                return ""
            except Exception as e:
                err_name = type(e).__name__
                status = getattr(e, "status_code", None)
                if err_name in ("RateLimitError", "APIStatusError") or status in (429, 529, 503):
                    last_err = e
                    wait = _retry_backoff(attempt)
                    logger.warning(
                        f"OpenAI rate/overload hit (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}). "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"OpenAI API error: {e}")
                return ""
        logger.error(f"Gave up after {MAX_RATE_LIMIT_RETRIES} retries (openai): {last_err}")
        return ""

    def check_connection(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection check failed: {e}")
            return False

    def check_model_exists(self) -> bool:
        return True


class GeminiClient:
    """Google Gemini API client via google-generativeai SDK.

    Token-savings features enabled by default:
      - Context caching: static system prompts are registered as CachedContent
        and reused across calls (one cache per unique system prompt hash).
      - Structured output: response_schema enforces JSON-only output matching
        either the message or action shape (auto-detected from system prompt).
    """

    CACHE_TTL = datetime.timedelta(hours=1)

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 200,
        enable_cache: bool = True,
        enable_structured_output: bool = True,
    ):
        import google.generativeai as genai
        self._genai = genai
        self.base_url = base_url or "https://generativelanguage.googleapis.com"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache
        self.enable_structured_output = enable_structured_output
        # hash(system_prompt) -> CachedContent or False (cache permanently disabled for this key)
        self._cache_handles: dict = {}
        self._cache_lock = threading.Lock()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY environment variable is not set")
            self._client = None
        else:
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model)

    def _get_or_create_cache(self, system_prompt: str):
        """Return a CachedContent for this system prompt, or None if caching is unavailable.

        Thread-safe double-checked creation: the lock only covers the create
        call, so cached-hit calls never contend on it.
        """
        key = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        handle = self._cache_handles.get(key, "missing")
        if handle != "missing":
            return handle if handle else None
        with self._cache_lock:
            handle = self._cache_handles.get(key, "missing")
            if handle != "missing":
                return handle if handle else None
            try:
                from google.generativeai import caching
                model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                cache = caching.CachedContent.create(
                    model=model_name,
                    system_instruction=system_prompt,
                    ttl=self.CACHE_TTL,
                )
                self._cache_handles[key] = cache
                logger.info(f"Gemini cache created: {cache.name} (prompt_hash={key[:12]})")
                return cache
            except Exception as e:
                logger.warning(
                    f"Gemini cache create failed (prompt_hash={key[:12]}); "
                    f"falling back to uncached path. Reason: {e}"
                )
                self._cache_handles[key] = False
                return None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._client is None:
            return ""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if self.enable_structured_output:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = _pick_gemini_schema(system_prompt)

        last_err = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                cache = self._get_or_create_cache(system_prompt) if self.enable_cache else None
                if cache is not None:
                    model = self._genai.GenerativeModel.from_cached_content(cached_content=cache)
                else:
                    model = self._genai.GenerativeModel(
                        self.model,
                        system_instruction=system_prompt,
                    )
                resp = model.generate_content(
                    user_prompt,
                    generation_config=gen_config,
                )
                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    logger.info(
                        "Token usage (gemini): input=%s, cache_read=%s, output=%s",
                        getattr(usage, "prompt_token_count", 0),
                        getattr(usage, "cached_content_token_count", 0),
                        getattr(usage, "candidates_token_count", 0),
                    )
                text = getattr(resp, "text", None)
                return text.strip() if text else ""
            except Exception as e:
                msg = str(e).lower()
                err_name = type(e).__name__
                if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg or err_name == "ResourceExhausted":
                    last_err = e
                    wait = _retry_backoff(attempt)
                    logger.warning(
                        f"Gemini rate/quota hit (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}). "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Gemini API error: {e}")
                return ""
        logger.error(f"Gave up after {MAX_RATE_LIMIT_RETRIES} retries (gemini): {last_err}")
        return ""

    def start_chat_session(self, system_instruction: str):
        """Create an isolated chat session with a fixed system_instruction.

        この session は state-ful: send_to_chat() を呼ぶ度に履歴 (user/model 交互) が
        session 内で蓄積される。moltbook 風の per-agent thread を作る用途。
        Returns the chat session object (Gemini SDK の ChatSession)."""
        if self._client is None:
            return None
        try:
            model = self._genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction,
            )
            return model.start_chat(history=[])
        except Exception as e:
            logger.warning(f"Gemini chat session create failed: {e}")
            return None

    def send_to_chat(
        self,
        chat,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        schema_kind: str = "auto",
    ) -> str:
        """既存 chat session に送信。history は session 内で自動蓄積される。

        schema_kind: "auto" (system_instruction を見て自動選択は不可なので message固定),
                     "message" / "message_with_memory" / "action" / "none"."""
        if chat is None:
            return ""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if self.enable_structured_output and schema_kind != "none":
            gen_config["response_mime_type"] = "application/json"
            if schema_kind == "message":
                gen_config["response_schema"] = _GEMINI_SCHEMA_MESSAGE
            elif schema_kind == "message_with_memory":
                gen_config["response_schema"] = _GEMINI_SCHEMA_MESSAGE_WITH_MEMORY
            elif schema_kind == "action":
                gen_config["response_schema"] = _GEMINI_SCHEMA_ACTION
            else:
                gen_config["response_schema"] = _GEMINI_SCHEMA_MESSAGE_WITH_MEMORY

        last_err = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                resp = chat.send_message(user_prompt, generation_config=gen_config)
                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    logger.info(
                        "Token usage (gemini-chat): input=%s, cache_read=%s, output=%s",
                        getattr(usage, "prompt_token_count", 0),
                        getattr(usage, "cached_content_token_count", 0),
                        getattr(usage, "candidates_token_count", 0),
                    )
                text = getattr(resp, "text", None)
                return text.strip() if text else ""
            except Exception as e:
                msg = str(e).lower()
                err_name = type(e).__name__
                if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg or err_name == "ResourceExhausted":
                    last_err = e
                    wait = _retry_backoff(attempt)
                    logger.warning(
                        f"Gemini-chat rate hit (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}). "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Gemini-chat error: {e}")
                return ""
        logger.error(f"Gave up after {MAX_RATE_LIMIT_RETRIES} retries (gemini-chat): {last_err}")
        return ""

    def check_connection(self) -> bool:
        if self._client is None:
            return False
        try:
            model = self._genai.GenerativeModel(self.model)
            model.generate_content("hi", generation_config={"max_output_tokens": 5})
            return True
        except Exception as e:
            logger.error(f"Gemini connection check failed: {e}")
            return False

    def check_model_exists(self) -> bool:
        return True


def create_llm_client(llm_config: dict):
    """Create LLM client based on provider in config.

    llm_config keys: provider (anthropic|openai|google), model, base_url,
    temperature, max_tokens. Defaults to anthropic for back-compat.
    """
    provider = (llm_config.get("provider") or "anthropic").lower()
    model = llm_config.get("model")
    base_url = llm_config.get("base_url")
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 200)

    if provider == "anthropic":
        from claude_client import ClaudeClient
        return ClaudeClient(
            base_url=base_url,
            model=model or "claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "openai":
        return OpenAIClient(
            base_url=base_url,
            model=model or "gpt-5.4-nano",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "google":
        return GeminiClient(
            base_url=base_url,
            model=model or "gemini-2.5-flash-lite",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
