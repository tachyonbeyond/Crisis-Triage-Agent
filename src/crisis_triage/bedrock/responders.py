"""Bedrock responders. Bedrock is the default path, not a fallback (docs/HACKATHON.md
section 7, pitfall 2) -- the templated text here is used only if Bedrock itself fails,
and it says so.

Three styles, one per route:

- CRISIS    : immediate resources, one concrete next step. Short by design.
- CHECK_IN  : one gentle, direct question; support line stays visible.
- CALM      : listens, offers options, never dismisses, never withholds a support line.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..resources import crisis_resources_text, full_crisis_block
from .client import BedrockError, BedrockTextClient

BASE_SAFETY = """You are a warm, plain-spoken crisis-support assistant. You are part of a
triage demo, not a substitute for emergency or professional services.

Hard rules:
- Never dismiss or minimise what the person said. Never tell them they are overreacting.
- Never promise confidentiality, and never claim to be human.
- Never give a diagnosis or medical advice.
- Keep replies short: 2-5 sentences. No lists of ten things.
- Use the person's own words where it helps them feel heard.
- Never include a phone number you were not given in the instructions."""

CRISIS_PROMPT = """{safety}

The safety check just classified this person as needing crisis resources now.

Write a short reply that:
1. Acknowledges what they said in one sentence, without alarm or judgement.
2. Gives ONE concrete next step they can take right now.
3. Includes these resources verbatim, exactly once:
{resources}

Do not ask them to explain themselves. Do not be cheerful. Do not bury the next step."""

CHECK_IN_PROMPT = """{safety}

The safety check did not find an immediate emergency, but the situation is unclear
enough that one direct question is warranted.

Write a short reply that:
1. Reflects what they said in one sentence.
2. Asks ONE gentle, direct, open question that would help tell whether they are safe
   right now -- for example whether they are thinking of acting on anything, or
   whether they are safe tonight. Do not make it sound like an interrogation.
3. Keeps this support line visible, at the end:
{resources}

Warm, not clinical. No advice yet, no reassurance that things will be fine."""

CALM_PROMPT = """{safety}

No immediate emergency was detected. This person is distressed but not in danger.

Write a short reply that:
1. Reflects what they said so they feel heard.
2. Offers one or two concrete options they could consider, phrased as options and not
   instructions.
3. Ends by mentioning that this support line is there if things get worse:
{resources}

Do not dismiss their distress, do not rush to fix it, and do not withhold the support
line. No directives."""


@dataclass(frozen=True)
class ResponderResult:
    text: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    used_fallback: bool = False
    error: str | None = None


class Responder:
    """Renders one reply for a given route via Bedrock."""

    def __init__(self, client: BedrockTextClient | None = None) -> None:
        self.client = client or BedrockTextClient()

    def _system(self) -> str:
        return BASE_SAFETY

    def respond(self, user_text: str, route_value: str, safety_note: str = "") -> ResponderResult:
        template = {
            "crisis": CRISIS_PROMPT,
            "check_in": CHECK_IN_PROMPT,
            "calm": CALM_PROMPT,
        }.get(route_value)
        if template is None:
            raise ValueError(f"unknown route {route_value!r}")

        system = template.format(
            safety=self._system()
            + (f"\n\nSafety-check note: {safety_note}" if safety_note else ""),
            resources=crisis_resources_text(),
        )
        try:
            completion = self.client.complete(user_text, system=system, max_tokens=400)
        except BedrockError as exc:
            return ResponderResult(
                text=self._fallback(route_value),
                latency_ms=float("nan"),
                used_fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ResponderResult(
            text=completion.text,
            latency_ms=completion.latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

    @staticmethod
    def _fallback(route_value: str) -> str:
        """Only reached if Bedrock is down. Never silent, never an all-clear."""
        if route_value == "calm":
            return (
                "I am here and I am listening, but my reply generator is unavailable "
                f"right now.\n\n{crisis_resources_text()}"
            )
        return full_crisis_block()


def crisis_responder(client: BedrockTextClient | None = None) -> Responder:
    return Responder(client)


def calm_converser(client: BedrockTextClient | None = None) -> Responder:
    return Responder(client)