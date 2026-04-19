"""
Anthropic Claude API client for LLM agent communication.
Supports prompt caching via (system_prompt, user_prompt) split.
"""
import os
import time
import logging
from typing import List, Optional

from anthropic import Anthropic, APIError, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 200
MAX_RATE_LIMIT_RETRIES = 4


class ClaudeClient:
    """Anthropic Claude API client with prompt caching."""

    def __init__(
        self,
        base_url: str = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.base_url = base_url or "https://api.anthropic.com"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY environment variable is not set")
            self._client = None
        else:
            self._client = Anthropic(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate response with prompt caching enabled.

        Args:
            system_prompt: Static part of the prompt (cached across calls).
            user_prompt: Dynamic part that changes per call.
        """
        if self._client is None:
            return ""

        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        last_rate_limit_err: Optional[Exception] = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )

                usage = getattr(msg, "usage", None)
                if usage is not None:
                    logger.info(
                        "Token usage: input=%s, cache_read=%s, cache_creation=%s, output=%s",
                        getattr(usage, "input_tokens", 0),
                        getattr(usage, "cache_read_input_tokens", 0),
                        getattr(usage, "cache_creation_input_tokens", 0),
                        getattr(usage, "output_tokens", 0),
                    )

                if msg.content and len(msg.content) > 0:
                    return msg.content[0].text.strip()
                return ""
            except RateLimitError as e:
                last_rate_limit_err = e
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status == 529 and attempt < MAX_RATE_LIMIT_RETRIES - 1:
                    # Overloaded — back off and retry.
                    wait = 2 ** attempt
                    logger.warning(
                        f"API overloaded (529). Retrying in {wait}s... ({attempt + 1}/{MAX_RATE_LIMIT_RETRIES})"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Claude API status error: {e}")
                return ""
            except APIError as e:
                logger.error(f"Error calling Claude API: {e}")
                return ""
            except Exception as e:
                logger.error(f"Unexpected error in Claude client: {e}")
                return ""
        logger.error(f"Gave up after {MAX_RATE_LIMIT_RETRIES} rate-limit retries: {last_rate_limit_err}")
        return ""

    def check_connection(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"Claude connection check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        return [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]

    def check_model_exists(self) -> bool:
        return True
