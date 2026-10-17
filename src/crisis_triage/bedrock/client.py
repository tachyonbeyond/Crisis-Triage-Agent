"""Shared Amazon Bedrock access for the safety check and the responders.

Auth is a Bedrock API key in `.env` as `AWS_BEARER_TOKEN_BEDROCK`. `.env` is loaded
here, before any client is constructed, because boto3 does not read `.env` itself.
The key is never read, printed, logged or echoed -- only its presence is checked.

The model ID is pinned rather than defaulted. The plain
`anthropic.claude-haiku-4-5-20251001-v1:0` is rejected for on-demand throughput and
must go through an inference profile; verified 2026-10-17 (docs/GROUND_TRUTH.md
section 6), so `.env` carries the working `us.`-prefixed form.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class BedrockError(RuntimeError):
    """Bedrock was unreachable, unauthorised, or returned something unusable."""


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _client_config():
    from botocore.config import Config

    # No retries by default: a slow safety check is worse than a reported failure,
    # and the caller has its own fail-safe.
    return Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 1})


class BedrockTextClient:
    """Minimal `converse` wrapper. Both the safety check and the responders use it."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        *,
        require_key: bool = True,
    ) -> None:
        load_dotenv()
        self.region = region or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
        self._has_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") is not None
        if require_key and not self._has_key:
            raise BedrockError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. Put it in .env (see .env.example); "
                "it must not be committed."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region, config=_client_config()
            )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        started = time.perf_counter()
        try:
            response = self.client.converse(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised as one type, message kept
            raise BedrockError(f"{type(exc).__name__}: {str(exc)[:400]}") from exc
        elapsed = (time.perf_counter() - started) * 1000.0

        blocks = response.get("output", {}).get("message", {}).get("content", []) or []
        text = "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict)
        ).strip()
        usage = response.get("usage", {}) or {}
        return Completion(
            text=text,
            latency_ms=elapsed,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            stop_reason=response.get("stopReason"),
            raw=response,
        )

    def complete_json(self, prompt: str, *, system: str, max_tokens: int = 200) -> tuple[dict, Completion]:
        """Ask for JSON and parse it, tolerating a fenced or chatty reply."""
        completion = self.complete(prompt, system=system, max_tokens=max_tokens)
        return extract_json_object(completion.text), completion

    def __repr__(self) -> str:
        return (
            f"BedrockTextClient(model_id={self.model_id!r}, region={self.region!r}, "
            f"key_present={self._has_key})"
        )


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or fences often enough that failing on that would be
    brittle; failing on genuinely unparseable text is correct.
    """
    if not text:
        raise BedrockError("model returned an empty response")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise BedrockError(f"no JSON object in model reply: {text[:200]!r}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise BedrockError(f"malformed JSON in model reply: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BedrockError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed