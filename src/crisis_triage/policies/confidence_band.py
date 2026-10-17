"""`ConfidenceBandPolicy` -- the reusable decision pattern (`#policy`).

Domain-agnostic: a typed `noul` question plus two calibrated thresholds decide a route.
Nothing here knows about crisis intervention; the crisis configuration is just one
instantiation (`crisis_policy.py`). Point it at "is this refund request fraudulent?"
and it works the same way.

    policy = ConfidenceBandPolicy(client, question="is_emergency", low=0.12, high=0.18)
    decision = policy.decide(user_turns)
    decision.route      # Route.CRISIS | CHECK_IN | CALM

Two things this module deliberately does NOT do:

1. It does not fail safe. On a Decider error it raises `DeciderError` -- the caller
   decides what an unreachable model means. In a crisis domain the answer is "show
   resources", and burying that choice in here would make it invisible.
2. It does not escalate automatically. The escalation signal is *reported*
   (`Decision.escalating_probability`) and the caller combines it. See
   `crisis_policy.py` for why: an OR of two individually-weak signals is what makes
   the numbers work out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from ..decider_client import DeciderClient, DeciderError
from .questions import CRISIS_QUESTIONS, DEFAULT_MAX_TURNS, ESCALATING, IS_EMERGENCY, build_state


class Route(str, Enum):
    CRISIS = "crisis"
    CHECK_IN = "check_in"
    CALM = "calm"


class BandPolicyError(RuntimeError):
    """Misconfiguration, or a Decider response that cannot be routed."""


@dataclass(frozen=True)
class Decision:
    """What one turn decided, with everything needed to explain it."""

    route: Route
    probability: float
    low: float
    high: float
    question: str
    escalating_probability: float | None = None
    # Set when routing happened without the model: an unreachable Decider, or a
    # response we could not parse. Always at least as cautious as the caller.
    fallback_reason: str | None = None
    crisis_type: str | None = None
    urgency: float | None = None
    decider_latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def explain(self) -> str:
        """One line for `--interactive`/`--demo` and for the logs."""
        if self.fallback_reason:
            return f"route={self.route.value.upper()} (fallback: {self.fallback_reason})"
        band = "HIGH" if self.probability >= self.high else ("LOW" if self.probability <= self.low else "MID")
        parts = [f"route={self.route.value.upper()}", f"P({self.question})={self.probability:.4f}"]
        if self.escalating_probability is not None:
            parts.append(f"P(escalating)={self.escalating_probability:.4f}")
        parts.append(
            f"bands=[{self.low:.2f},{self.high:.2f}] {band}"
            + (f" latency={self.decider_latency_ms:.0f}ms" if self.decider_latency_ms else "")
        )
        return "  ".join(parts)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise BandPolicyError(f"{name}={raw!r} is not a number") from exc


class ConfidenceBandPolicy:
    """Route on where a calibrated probability falls relative to two thresholds.

    `low > high` is rejected rather than silently producing a policy that can never
    route to CRISIS.
    """

    def __init__(
        self,
        client: DeciderClient,
        *,
        question: str = IS_EMERGENCY,
        low: float | None = None,
        high: float | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        questions: dict[str, dict] | None = None,
        escalation_question: str | None = ESCALATING,
    ) -> None:
        resolved_low = low if low is not None else _env_float("CRISIS_LOW")
        resolved_high = high if high is not None else _env_float("CRISIS_HIGH")
        if resolved_low is None or resolved_high is None:
            raise BandPolicyError(
                "thresholds not set: pass low= and high=, or set CRISIS_LOW and "
                "CRISIS_HIGH in the environment. They are intentionally not defaulted "
                "-- the values must come from a measured sweep (see "
                "docs/GROUND_TRUTH.md section 4)."
            )
        if not 0.0 <= resolved_low <= resolved_high <= 1.0:
            raise BandPolicyError(
                f"need 0 <= low <= high <= 1, got low={resolved_low}, high={resolved_high}"
            )

        self.client = client
        self.question = question
        self.low = resolved_low
        self.high = resolved_high
        self.max_turns = max_turns
        self.questions = questions if questions is not None else CRISIS_QUESTIONS
        self.escalation_question = escalation_question

        if question not in self.questions:
            raise BandPolicyError(f"question {question!r} is not in the question set")

    # -- routing -----------------------------------------------------------

    def route_for(self, probability: float) -> Route:
        """The band rule, isolated so callers and tests can apply it to any value."""
        if probability >= self.high:
            return Route.CRISIS
        if probability <= self.low:
            return Route.CALM
        return Route.CHECK_IN

    def decide(
        self,
        user_turns: list[str],
        *,
        questions: dict[str, dict] | None = None,
    ) -> Decision:
        """Score the turns and route. Raises `DeciderError` if the model is unusable."""
        state = build_state(user_turns, self.max_turns)
        asked = questions if questions is not None else self.questions
        response = self.client.evaluate(state, asked)

        try:
            probability = response.noul(self.question)
        except DeciderError as exc:
            raise BandPolicyError(
                f"cannot route: {exc}. The question set must include a `noul` question "
                f"named {self.question!r}."
            ) from exc

        escalating: float | None = None
        if self.escalation_question and self.escalation_question in response.answers:
            escalating = response.noul(self.escalation_question)

        crisis_type = urgency = None
        if "crisis_type" in response.answers:
            crisis_type = response.choice("crisis_type").choice
        if "urgency_level" in response.answers:
            urgency = response.score("urgency_level").score

        return Decision(
            route=self.route_for(probability),
            probability=probability,
            low=self.low,
            high=self.high,
            question=self.question,
            escalating_probability=escalating,
            crisis_type=crisis_type,
            urgency=urgency,
            decider_latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    def decide_or_fallback(self, user_turns: list[str]) -> Decision:
        """`decide`, but a Decider failure yields a CRISIS-routed Decision instead of raising.

        Fail safe in the only direction that is acceptable here: toward showing
        resources, never toward silence (docs/DECIDER.md section 4, rule 5).
        """
        try:
            return self.decide(user_turns)
        except (DeciderError, BandPolicyError) as exc:
            return Decision(
                route=Route.CRISIS,
                probability=float("nan"),
                low=self.low,
                high=self.high,
                question=self.question,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

    def __repr__(self) -> str:
        return (
            f"ConfidenceBandPolicy(question={self.question!r}, "
            f"low={self.low}, high={self.high}, max_turns={self.max_turns})"
        )