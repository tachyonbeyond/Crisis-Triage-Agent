"""Crisis resources.

Provenance, per docs/DECIDER.md section 7: every number below was read from the
operator's own page on 2026-10-17, not written from memory. Re-verify before
publishing anything.

- 988 Suicide & Crisis Lifeline -- https://988lifeline.org/ : call or text **988**,
  chat at chat.988lifeline.org, Spanish text/chat available, free and confidential,
  described by the operator as a national 24/7/365 service.
- **911** for immediate danger or a medical emergency.

These are US-centric. That is a real limitation of this demo and is stated in the
README rather than papered over.
"""

from __future__ import annotations

US_LIFELINE = "988"
US_EMERGENCY = "911"
US_CHAT = "chat.988lifeline.org"


def crisis_resources_text() -> str:
    """The support line shown on every path that could involve risk."""
    return (
        f"If you are in immediate danger, call {US_EMERGENCY} now. "
        f"In the US you can call or text {US_LIFELINE} any time, free and confidential, "
        f"or chat at {US_CHAT}. Spanish-language text and chat are available."
    )


def unavailable_notice() -> str:
    """Shown when the automatic check itself could not run.

    Says plainly that the automatic check failed, so nobody reads silence as "safe".
    """
    return (
        "I could not run the automatic safety check just now, so please do not treat "
        "this reply as an all-clear."
    )


def full_crisis_block() -> str:
    return f"{unavailable_notice()}\n\n{crisis_resources_text()}"