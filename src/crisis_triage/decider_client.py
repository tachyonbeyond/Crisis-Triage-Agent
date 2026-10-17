"""Client for a locally served Strands Decider 2B (`POST /v1/systemone`).

Shapes here were confirmed against a live server and the installed package on
2026-10-17 -- see docs/GROUND_TRUTH.md section 2. In particular:
`answers.<name>.noul` is real, `latency_ms` is injected by the server (not part of
the pydantic response), and `choice` carries no per-option confidence.

The client is deliberately dumb: it returns what the server said and raises
`DeciderError` when the server did not say anything usable. It does **not** decide
what an unreachable model means -- that is the policy layer's job, and in a crisis
domain getting it wrong means failing silently (docs/DECIDER.md section 4).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 30.0


class DeciderError(RuntimeError):
    """Decider was unreachable, timed out, or returned something unusable."""


@dataclass(frozen=True)
class NoulAnswer:
    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class DeciderResponse:
    """One call's answers plus the cost/latency bookkeeping we benchmark on."""

    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def noul(self, name: str) -> float:
        """P(true) for a `noul` question. Raises if the question was a different type."""
        answer = self._require(name, NoulAnswer)
        return answer.probability

    def choice(self, name: str) -> ChoiceAnswer:
        return self._require(name, ChoiceAnswer)

    def score(self, name: str) -> ScoreAnswer:
        return self._require(name, ScoreAnswer)

    def _require(self, name: str, kind: type):
        if name not in self.answers:
            raise DeciderError(f"no answer named {name!r}; got {sorted(self.answers)}")
        answer = self.answers[name]
        if not isinstance(answer, kind):
            raise DeciderError(
                f"answer {name!r} is {type(answer).__name__}, expected {kind.__name__}"
            )
        return answer


def _parse_answer(name: str, payload: dict[str, Any]):
    """Turn one entry of the response's `answers` into a typed answer.

    Unknown shapes raise rather than guess: a silently misread probability in this
    domain is worse than a loud failure.
    """
    kind = payload.get("type")
    if kind == "noul":
        return NoulAnswer(probability=float(payload["noul"]))
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(payload["choice"]),
            probabilities={k: float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(payload["score"]),
            legend={str(k): str(v) for k, v in payload["legend"].items()},
            probabilities={str(k): float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    raise DeciderError(f"answer {name!r} has unknown type {kind!r}")


class DeciderClient:
    """Thin HTTP client for the local Decider server.

    All questions for one turn are sent in a **single** request: the server encodes
    the state once and shares its KV prefix across the batch, so extra questions are
    nearly free (strands_decider/infer.py:360).
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("DECIDER_URL") or DEFAULT_URL).rstrip("/")
        if timeout_s is None:
            timeout_s = float(os.environ.get("DECIDER_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
        self.timeout_s = timeout_s

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout_s) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001 - detail is best-effort
                pass
            raise DeciderError(f"HTTP {exc.code} from {path}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise DeciderError(f"cannot reach Decider at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise DeciderError(f"Decider timed out after {timeout or self.timeout_s}s") from exc
        except json.JSONDecodeError as exc:
            raise DeciderError(f"Decider returned non-JSON body: {exc}") from exc

    # -- api ---------------------------------------------------------------

    def health(self, timeout_s: float = 10.0) -> dict[str, Any]:
        """`GET /health` -- verified to exist on 2026-10-17."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout_s) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - surfaced as one error type
            raise DeciderError(f"Decider health check failed: {exc!r}") from exc

    def is_healthy(self, timeout_s: float = 10.0) -> bool:
        try:
            return bool(self.health(timeout_s=timeout_s).get("status") == "ok")
        except DeciderError:
            return False

    def evaluate(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
        timeout_s: float | None = None,
    ) -> DeciderResponse:
        """Score `questions` against `state` in one batched call."""
        if not questions:
            raise DeciderError("at least one question is required")
        if not state.strip():
            raise DeciderError("state must not be empty (the server rejects it too)")

        payload = self._post("/v1/systemone", {"state": state, "questions": questions}, timeout_s)

        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, dict) or not raw_answers:
            raise DeciderError(f"response has no usable 'answers': {payload!r}"[:500])

        try:
            answers = {name: _parse_answer(name, body) for name, body in raw_answers.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise DeciderError(f"could not parse answers: {exc!r}") from exc

        usage = payload.get("usage") or {}
        latency = payload.get("latency_ms")
        return DeciderResponse(
            answers=answers,
            model=str(payload.get("model", "unknown")),
            latency_ms=float(latency) if latency is not None else float("nan"),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            raw=payload,
        )

    def __repr__(self) -> str:
        return f"DeciderClient(base_url={self.base_url!r}, timeout_s={self.timeout_s})"