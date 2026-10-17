"""Crisis configuration of the band pattern, and its Strands wiring.

## The routing rule, and why it is an OR

Measured on 22 fixtures (docs/GROUND_TRUTH.md section 4): Decider's own
`is_emergency` probability does not separate crisis from non-crisis by any single
threshold. A stated plan with means scored 0.2158; a check-in case reached 0.1765.
So Decider alone cannot be the thing that decides whether someone gets crisis
resources -- it decides *tone*, which it is reliable at and does in ~165 ms.

Safety is therefore a three-signal OR, and no emergency depends on Decider alone:

    safety.emergency                     # Bedrock classifier -- got 7/7 right
    or band_decision.route is CRISIS     # Decider's high band
    or escalating_probability >= ESC     # Decider's escalation signal
    or band_decision.fallback_reason     # Decider unreachable -> fail safe

Any one of them routes to CRISIS. That is the whole point of the design: two
individually-weak signals OR'd together, plus one strong one.

## Wiring note

The handler hangs off `before_invocation` for the once-per-turn triage and injects
its directive from `before_model_call` via `Guide`. That split is not arbitrary:
per the installed SDK, `Guide` returned from `before_invocation` *cancels* the
invocation, while `Guide` from `before_model_call` appends the feedback as a user
message and lets the model proceed. See docs/GROUND_TRUTH.md section 7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ..bedrock.safety_check import SafetyChecker, SafetyVerdict
from .confidence_band import BandPolicyError, ConfidenceBandPolicy, Decision, Route

DEFAULT_ESC_THRESHOLD = 0.70

# Key under which the triage outcome is parked in `event.invocation_state`.
_INVOCATION_KEY = "crisis_triage"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class TriageOutcome:
    """Everything one turn decided, with a full audit trail for the demo."""

    route: Route
    band: Decision
    safety: SafetyVerdict | None = None
    triggered_by: list[str] = field(default_factory=list)

    @property
    def guidance(self) -> str:
        """The directive injected into the agent's context. Internal, not user-facing.

        The support line is appended by code after generation, so these directives must
        NOT ask the model to produce it -- and the calm one must actively keep crisis
        resources out of an ordinary reply.
        """
        if self.route is Route.CRISIS:
            return (
                "SAFETY ROUTING: this turn needs an immediate safety response. Stay with the "
                "person: acknowledge in one neutral sentence without repeating what they "
                "described, ask directly whether they are safe right now, and give exactly ONE "
                "concrete step they can take immediately (for example putting distance between "
                "themselves and what they mentioned, and contacting help). Three or four "
                "sentences, no lists. Never introduce or describe methods, quantities or "
                "effects; if they mentioned one, refer to it only as 'what you mentioned'. "
                "Do not ask them to explain themselves and do not be cheerful. Do not add a "
                "phone number or link: the support line is appended after your reply."
            )
        if self.route is Route.CHECK_IN:
            return (
                "SAFETY ROUTING: no immediate emergency was detected, but the situation is "
                "unclear. Reply warmly in two or three sentences, then ask exactly ONE gentle, "
                "direct question about whether they are safe right now -- for example whether "
                "they are having thoughts of hurting themselves, or whether they are safe "
                "tonight. Do not give advice yet. Do not add a phone number or link: the support "
                "line is appended after your reply."
            )
        return (
            "SAFETY ROUTING: no immediate emergency was detected and this reads as an ordinary "
            "hard day. Reply warmly in two or three sentences: reflect what they said and offer "
            "one or two practical options, phrased as options rather than instructions. Do not "
            "mention crisis resources, hotlines, the emergency number or links -- they are "
            "appended by code only when the message calls for them. Do not dismiss their day "
            "and do not moralise."
        )

    def explain(self) -> str:
        """One audit line for the demo. Reports the EFFECTIVE route first.

        Reporting the band's route here was a real bug: when the safety check raised
        the route, the demo still printed the band's verdict, so a crisis turn read as
        `route=CALM`. The effective route is the one that drove the reply.
        """
        band_route = self.band.route.value
        bits = [f"route={self.route.value.upper()}"]
        if self.route is not self.band.route:
            bits.append(f"(band said {band_route.upper()})")
        bits.append(self.band.explain())
        if self.safety is not None:
            bits.append(f"bedrock_safety={self.safety.emergency}")
        if self.triggered_by:
            bits.append(f"raised_by={'+'.join(self.triggered_by)}")
        return "  ".join(bits)


class CrisisTriagePolicy:
    """Combines Decider's tone/escalation reading with Bedrock's safety verdict."""

    def __init__(
        self,
        band_policy: ConfidenceBandPolicy,
        safety_checker: SafetyChecker | None = None,
        *,
        esc_threshold: float | None = None,
    ) -> None:
        self.band_policy = band_policy
        self.safety_checker = safety_checker
        self.esc_threshold = (
            esc_threshold if esc_threshold is not None else _env_float("CRISIS_ESC", DEFAULT_ESC_THRESHOLD)
        )

    def triage(self, user_turns: list[str]) -> TriageOutcome:
        state = "\n".join(f"USER: {t}" for t in user_turns[-self.band_policy.max_turns :])
        return self.triage_state(state, user_turns)

    def triage_state(self, state: str, user_turns: list[str] | None = None) -> TriageOutcome:
        """Triage an explicit state string (used by the handler, which has the messages)."""
        turns = user_turns or [state]
        band = self.band_policy.decide_or_fallback(turns)

        safety = self.safety_checker.check(state) if self.safety_checker is not None else None

        triggered: list[str] = []
        if band.fallback_reason:
            triggered.append("decider_unavailable")
        if band.route is Route.CRISIS:
            triggered.append("decider_high_band")
        esc = band.escalating_probability
        if esc is not None and esc >= self.esc_threshold:
            triggered.append("decider_escalating")
        if safety is not None and safety.emergency:
            triggered.append("bedrock_safety")

        route = Route.CRISIS if triggered else band.route
        return TriageOutcome(route=route, band=band, safety=safety, triggered_by=triggered)

    def __repr__(self) -> str:
        return (
            f"CrisisTriagePolicy(band_policy={self.band_policy!r}, "
            f"esc_threshold={self.esc_threshold}, safety={'on' if self.safety_checker else 'off'})"
        )


# ---------------------------------------------------------------------------
# Strands wiring
# ---------------------------------------------------------------------------


def _message_texts(messages: Any) -> list[str]:
    """Extract user text from a Strands message list, tolerating both known shapes."""
    out: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
            continue
        chunks: list[str] = []
        for block in content or []:
            if isinstance(block, dict) and "text" in block:
                chunks.append(str(block["text"]))
        if chunks:
            out.append(" ".join(chunks))
    return out


def _make_handler_class():
    """Build the handler class with the real SDK base, imported lazily.

    Importing `strands` at module import time would make this module unusable in a
    test run that has no Strands installed, and the handler is one of two consumers.
    """
    from strands.interventions import Guide, InterventionHandler, Proceed

    class CrisisInterventionHandler(InterventionHandler):
        """Runs triage once per user turn and steers the reply by route."""

        def __init__(
            self,
            policy: CrisisTriagePolicy,
            *,
            name: str = "crisis_triage",
        ) -> None:
            self.policy = policy
            self._name = name
            # The outcome this handler actually used, so callers can report the real
            # decision instead of recomputing a second one that might differ.
            self.last_outcome: TriageOutcome | None = None

        @property
        def name(self) -> str:  # required by InterventionHandler
            return self._name

        def before_invocation(self, event):
            """Triage once per turn and park the outcome in the invocation state."""
            messages = getattr(event, "messages", None)
            if messages is None:
                messages = getattr(getattr(event, "agent", None), "messages", None)
            turns = _message_texts(messages)
            if not turns:
                return Proceed()

            state = "\n".join(f"USER: {t}" for t in turns[-self.policy.band_policy.max_turns :])
            outcome = self.policy.triage_state(state, turns)
            self.last_outcome = outcome
            event.invocation_state[_INVOCATION_KEY] = outcome
            return Proceed()

        def before_model_call(self, event):
            """Inject the routing directive exactly once per turn.

            Injected on every model call it would accumulate in the transcript and
            fight itself, so the outcome is cleared after use.
            """
            outcome = event.invocation_state.pop(_INVOCATION_KEY, None)
            if outcome is None:
                return Proceed()
            return Guide(feedback=outcome.guidance)

    return CrisisInterventionHandler


_HANDLER_CLASS = None


def crisis_intervention_handler(policy: CrisisTriagePolicy, **kwargs):
    """Factory: `Agent(interventions=[crisis_intervention_handler(policy)])`."""
    global _HANDLER_CLASS
    if _HANDLER_CLASS is None:
        _HANDLER_CLASS = _make_handler_class()
    return _HANDLER_CLASS(policy, **kwargs)


__all__ = [
    "CrisisTriagePolicy",
    "TriageOutcome",
    "crisis_intervention_handler",
    "BandPolicyError",
]