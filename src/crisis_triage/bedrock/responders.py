"""Bedrock responders. Bedrock is the default path, not a fallback -- the templated
text here is used only if Bedrock itself fails, and it says so.

Three styles, one per route:

- CRISIS    : stays with the person, asks whether they are safe right now, one concrete
              step. Short by design and neutral about what they described.
- CHECK_IN  : one gentle, direct safety question.
- CALM      : listens, offers options, and does not raise crisis resources at all.

**The support line is not in these prompts.** It is appended in code after generation
(`resources.append_support_line`) from one constant per region, so its wording, presence
and placement never depend on the model. A model may paraphrase or omit instructions;
code cannot.

The prompts deliberately never ask the model to restate what the person described in
detail. In the crisis route that is a safety rule: naming or elaborating a method,
quantity or effect in a reply is exactly what we must not do.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..resources import append_support_line, full_crisis_block
from .client import BedrockError, BedrockTextClient

BASE_SAFETY = """You are a warm, plain-spoken crisis-support assistant. You are part of a
triage demo, not a substitute for emergency or professional services.

Hard rules:
- Never dismiss or minimise what the person said. Never tell them they are overreacting.
- Never promise confidentiality, and never claim to be human.
- Never give a diagnosis or medical advice.
- Keep replies short: 2-5 sentences. No lists of ten things.
- Use the person's own words only in neutral, general terms where it helps them feel heard.
- Never introduce or describe methods, quantities, dosages, or effects. If the person
  already described one, refer to it only as "what you mentioned".
- Never include a phone number, hotline, or link: they are added separately, in code."""

CRISIS_PROMPT = """{safety}

The safety check classified this turn as needing an immediate safety response.

Write a short reply that:
1. Stays with the person: one sentence that acknowledges they are going through something
   hard, in neutral terms. Do not repeat or characterise what they described in detail.
2. Asks directly whether they are safe right now.
3. Gives exactly ONE concrete step they can take immediately -- for example putting some
   distance between themselves and what they mentioned, and contacting help.
4. Does not ask them to explain themselves, and is not cheerful.

Three or four sentences. No lists. Do not name or describe any method, quantity or effect."""

CHECK_IN_PROMPT = """{safety}

The safety check did not find an immediate emergency, but the situation is unclear
enough that one direct question is warranted.

Write a short reply that:
1. Reflects what they said in one sentence.
2. Asks ONE gentle, direct question about whether they are safe right now -- for example
   whether they are having thoughts of hurting themselves, or whether they are safe
   tonight. Do not make it sound like an interrogation.

Two or three sentences. Warm, not clinical. No advice yet, and do not say things will be fine."""

CALM_PROMPT = """{safety}

No immediate emergency was detected. This person is dealing with an ordinary hard day,
not in danger.

Write a short reply that:
1. Reflects what they said so they feel heard.
2. Offers one or two practical options, phrased as options and not instructions.

Two or three sentences. Do not mention crisis resources, hotlines, the emergency number,
or links -- they are added separately, in code, only when the message calls for them.
Do not dismiss their day and do not moralise."""


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
            + (f"\n\nSafety-check note: {safety_note}" if safety_note else "")
        )
        try:
            completion = self.client.complete(user_text, system=system, max_tokens=400)
        except BedrockError as exc:
            return ResponderResult(
                text=self._fallback(route_value, user_text),
                latency_ms=float("nan"),
                used_fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ResponderResult(
            # The support line is applied to generated text too: the model is not asked
            # to produce it, so its presence is guaranteed here, not hoped for.
            text=append_support_line(completion.text, route_value, user_text),
            latency_ms=completion.latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

    @staticmethod
    def _fallback(route_value: str, user_text: str = "") -> str:
        """Only reached if Bedrock is down. Never silent, and always shows resources.

        Every route shows the line here, including calm: a failed generator means we
        could not tailor the reply, so we do not also withhold support.
        """
        return full_crisis_block()


def crisis_responder(client: BedrockTextClient | None = None) -> Responder:
    return Responder(client)


def calm_converser(client: BedrockTextClient | None = None) -> Responder:
    return Responder(client)