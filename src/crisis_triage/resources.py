"""Crisis resources.

The support line is built **in code** from one constant per region, never from model
wording. A support line that only appears when a language model chooses to emit it is
not a safety guarantee, so the reply prompts no longer mention it at all: the line is
appended after generation by `append_support_line()`.

Provenance of the US numbers: read from https://988lifeline.org/ on 2026-10-17 (call or
text **988**; chat at chat.988lifeline.org; Spanish text/chat available; free and
confidential). Re-verify before publishing.

Region selection: `SUPPORT_REGION` (default `"US"`). `US` shows 911 + 988 + the chat
link; any other value shows the local-wording fallback, because we have not verified
numbers for any other country and will not invent them.
"""

from __future__ import annotations

import os

SUPPORT_REGION_ENV = "SUPPORT_REGION"
DEFAULT_REGION = "US"

US_LIFELINE = "988"
US_EMERGENCY = "911"
US_CHAT = "chat.988lifeline.org"

US_SUPPORT_LINE = (
    f"If you are in immediate danger, call {US_EMERGENCY}. "
    f"You can also call or text {US_LIFELINE} any time, free and confidential, "
    f"or chat at {US_CHAT}. Spanish-language text and chat are available."
)

# Deliberately names no number: we have not verified hotlines outside the US.
LOCAL_SUPPORT_LINE = (
    "If you are in immediate danger, contact your local emergency number. "
    "You can also contact a local crisis line for support."
)

# Markers used to detect a line the model produced itself, so we never duplicate one.
_US_MARKERS = (US_LIFELINE, US_EMERGENCY, US_CHAT)

# Distress signals used ONLY to decide whether the calm route shows the support line.
# This is a small, deterministic list, not a classifier: it never affects routing, and
# it is intentionally conservative. It will miss unfamiliar phrasing -- on the calm
# route a miss means no banner, which is the one place this project accepts that risk.
_DISTRESS_PHRASES = (
    "hopeless",
    "worthless",
    "no point",
    "can't go on",
    "cant go on",
    "cannot go on",
    "can not go on",
    "don't know how much longer",
    "do not know how much longer",
    "give up",
    "panic",  # also matches "panicking"
    "lonely",
    "alone",
    "grief",
    "grieving",
    "miss her",
    "miss him",
    "miss them",
    "ashamed",
    "hate myself",
    "hurt myself",
    "kill myself",
    "end it",
    "suicid",
    "self-harm",
    "self harm",
    "depress",
    "anxious",
    "overwhelmed",
    "broken",
    "worthless",
    "despair",
    "crying",
    "cry for an hour",
    "help me",
    "someone to talk to",
    "need to talk",
    # Not the bare word "support": it matches ordinary complaints such as
    # "support will not answer", which is exactly the false positive the calm rule
    # exists to avoid. Only explicit requests count.
    "need support",
    "need some support",
    "support line",
    "reach out for support",
)


def support_line(region: str | None = None) -> str:
    """The support line for `region` (default: `SUPPORT_REGION`, else US).

    Unknown or non-US regions get the local wording rather than an invented number.
    """
    resolved = (
        region if region is not None else os.environ.get(SUPPORT_REGION_ENV, DEFAULT_REGION)
    )
    resolved = (resolved or DEFAULT_REGION).strip().upper()
    return US_SUPPORT_LINE if resolved == DEFAULT_REGION else LOCAL_SUPPORT_LINE


def crisis_resources_text(region: str | None = None) -> str:
    """Backwards-compatible alias for :func:`support_line`."""
    return support_line(region)


def has_distress_signal(text: str) -> bool:
    """Does the user's message read as distress or a request for support?

    Deterministic substring matching, lower-cased. Used only by the calm route.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _DISTRESS_PHRASES)


def shows_support_line(route: str, user_text: str = "") -> bool:
    """Should this reply carry the support line?

    check-in and crisis: always. calm: only when the message shows distress or asks for
    support, so an ordinary complaint does not get a crisis banner.
    """
    route_value = getattr(route, "value", route)
    if route_value in {"check_in", "crisis"}:
        return True
    if route_value == "calm":
        return has_distress_signal(user_text)
    return False


def append_support_line(
    text: str,
    route: str,
    user_text: str = "",
    region: str | None = None,
) -> str:
    """Append the code-built support line when the route calls for it.

    Idempotent: if the text already carries the line (the model produced one itself),
    it is left alone rather than duplicated.
    """
    if not shows_support_line(route, user_text):
        return text
    line = support_line(region)
    if line in text:
        return text
    # A model-written US-style line counts as present too; don't stack two.
    if support_line(region) == US_SUPPORT_LINE and any(m in text for m in _US_MARKERS):
        return text
    return f"{text.rstrip()}\n\n{line}"


def unavailable_notice() -> str:
    """Shown when the automatic check itself could not run.

    Says plainly that the automatic check failed, so nobody reads silence as "safe".
    """
    return (
        "I could not run the automatic safety check just now, so please do not treat "
        "this reply as an all-clear."
    )


def full_crisis_block(region: str | None = None) -> str:
    return f"{unavailable_notice()}\n\n{support_line(region)}"