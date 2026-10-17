"""The Bedrock safety check that backs the routing decision.

Why this exists at all: Decider 2B, measured on our own 22 fixtures, does not
separate crisis from non-crisis in a way any single threshold can act on. It scored
a stated suicide plan with means at P=0.2158 while a check-in case reached 0.1765
(docs/GROUND_TRUTH.md section 4). A Bedrock classifier put all 6 clear emergencies
and the passive-ideation control right in the same test (section 5).

So the safety question -- "does this person need crisis resources right now?" -- is
answered by Bedrock, and Decider answers the tone question. See policies/crisis_policy.py
for how the two are combined.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .client import BedrockError, BedrockTextClient

SAFETY_SYSTEM_PROMPT = """You are a crisis-safety classifier for a triage system.

You decide ONE thing: whether this person needs crisis resources right now.

Answer with a JSON object and nothing else:
{"emergency": true|false, "reason": "<max 15 words>"}

emergency = true when ANY of these hold:
- Imminent danger to self or others.
- An active suicide plan, or stated intent, means, or a timeline.
- A medical emergency needing 911 (overdose, severe injury, chest pain, stroke signs).
- Active violence, or a credible threat of violence to someone.
- Someone else reports a person is in immediate danger.

emergency = false when the person is distressed but safe: venting, anxiety, grief,
loneliness, burnout, past self-harm with no current intent, or passive thoughts
with no plan, means, or intent.

Err toward true when a real person could be harmed. A false alarm sends someone a
support line; a miss can kill someone."""


@dataclass(frozen=True)
class SafetyVerdict:
    emergency: bool
    reason: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    failed: bool = False


class SafetyChecker:
    """One Bedrock call per user turn answering "needs crisis resources now?"."""

    def __init__(self, client: BedrockTextClient | None = None) -> None:
        self.client = client or BedrockTextClient()
        # A failed safety check must not be treated as "safe".
        self.fail_closed = os.environ.get("SAFETY_FAIL_CLOSED", "1") not in {"0", "false", "False"}

    def check(self, text: str) -> SafetyVerdict:
        """Classify one state. On failure, returns `emergency=True` when fail-closed.

        Never raises for a model/transport problem: routing must always get an answer
        (docs/DECIDER.md section 4, rule 5).
        """
        try:
            parsed, completion = self.client.complete_json(
                text, system=SAFETY_SYSTEM_PROMPT, max_tokens=100
            )
        except BedrockError as exc:
            return SafetyVerdict(
                emergency=self.fail_closed,
                reason=f"safety check unavailable: {type(exc).__name__}",
                latency_ms=float("nan"),
                failed=True,
            )

        verdict = parsed.get("emergency")
        if not isinstance(verdict, bool):
            return SafetyVerdict(
                emergency=self.fail_closed,
                reason=f"unparseable verdict {verdict!r}",
                latency_ms=completion.latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                failed=True,
            )
        return SafetyVerdict(
            emergency=verdict,
            reason=str(parsed.get("reason", ""))[:200],
            latency_ms=completion.latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

    def __repr__(self) -> str:
        return f"SafetyChecker(fail_closed={self.fail_closed}, client={self.client!r})"