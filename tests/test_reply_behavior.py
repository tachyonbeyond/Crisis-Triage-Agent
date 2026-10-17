"""Reply behaviour: the support line, the banner rule, and the region switch.

The support line is a safety guarantee, so these tests assert it is built in code and
appears on every path that could involve risk -- including when a backend fails. They
also pin the calm rule: an ordinary complaint must NOT get a crisis banner.
"""

from __future__ import annotations

import pytest

from crisis_triage.bedrock.client import BedrockError
from crisis_triage.bedrock.responders import Responder
from crisis_triage.resources import (
    LOCAL_SUPPORT_LINE,
    US_SUPPORT_LINE,
    append_support_line,
    has_distress_signal,
    shows_support_line,
    support_line,
)

ROUTING_COMPLAINT = "My payouts have been failing for 3 days and support will not answer."
DISTRESSED = "Sometimes I feel completely hopeless and I do not know how much longer I can go on."
CRISIS_TEXT = "I decided tonight is the night."


# -- the line itself -------------------------------------------------------


def test_us_line_carries_the_verified_numbers():
    line = support_line("US")
    assert line == US_SUPPORT_LINE
    assert "911" in line
    assert "988" in line
    assert "chat.988lifeline.org" in line


@pytest.mark.parametrize("region", ["GB", "DE", "AU", "us-unknown"])
def test_non_us_regions_get_local_wording_and_no_invented_numbers(region):
    line = support_line(region)
    assert line == LOCAL_SUPPORT_LINE
    assert "local emergency number" in line
    assert "911" not in line
    assert "988" not in line


def test_region_defaults_to_us_and_can_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("SUPPORT_REGION", raising=False)
    assert support_line() == US_SUPPORT_LINE
    monkeypatch.setenv("SUPPORT_REGION", "gb")
    assert support_line() == LOCAL_SUPPORT_LINE
    # An explicit argument beats the environment.
    assert support_line("US") == US_SUPPORT_LINE


def test_region_is_case_insensitive_and_trimmed():
    assert support_line("  us  ") == US_SUPPORT_LINE


# -- the banner rule -------------------------------------------------------


@pytest.mark.parametrize("route", ["check_in", "crisis"])
def test_check_in_and_crisis_always_show_the_line(route):
    assert shows_support_line(route, ROUTING_COMPLAINT) is True
    assert support_line() in append_support_line("a reply", route, ROUTING_COMPLAINT)


def test_calm_without_distress_omits_the_line():
    out = append_support_line("a calm reply", "calm", ROUTING_COMPLAINT)
    assert out == "a calm reply"
    for marker in ("988", "911", "chat.988lifeline.org", "local emergency number"):
        assert marker not in out


def test_calm_with_distress_shows_the_line():
    assert has_distress_signal(DISTRESSED)
    assert support_line() in append_support_line("a reply", "calm", DISTRESSED)


@pytest.mark.parametrize(
    "text",
    [
        "I am panicking about the exam",
        "I feel worthless",
        "I just need someone to talk to",
        "my grief is overwhelming",
    ],
)
def test_distress_phrasings_are_recognised(text):
    assert has_distress_signal(text) is True


def test_appending_is_idempotent():
    once = append_support_line("reply", "crisis", CRISIS_TEXT)
    assert append_support_line(once, "crisis", CRISIS_TEXT) == once


def test_a_model_written_us_line_is_not_duplicated():
    model_text = "I'm here. Call 988 if you need support."
    out = append_support_line(model_text, "crisis", CRISIS_TEXT)
    assert out == model_text


def test_region_switch_applies_to_the_appended_line():
    assert LOCAL_SUPPORT_LINE in append_support_line("r", "crisis", CRISIS_TEXT, region="GB")


# -- failures still show resources ----------------------------------------


class _ExplodingClient:
    """A Bedrock client that always fails, to exercise the responder fallback."""

    model_id = "fake"
    region = "us-east-1"

    def complete(self, prompt, *, system=None, max_tokens=400, temperature=0.0):
        raise BedrockError("throttled by the service")


@pytest.mark.parametrize("route", ["calm", "check_in", "crisis"])
def test_bedrock_failure_still_shows_the_support_line(route):
    responder = Responder(_ExplodingClient())
    result = responder.respond("hello", route)
    assert result.used_fallback is True
    assert US_SUPPORT_LINE in result.text


def test_bedrock_failure_on_crisis_also_says_the_check_was_unavailable():
    result = Responder(_ExplodingClient()).respond(CRISIS_TEXT, "crisis")
    assert "could not run the automatic safety check" in result.text


class _CapturingClient:
    """Records the system prompt so the tests can prove the line is not model-generated."""

    model_id = "fake"
    region = "us-east-1"

    def __init__(self) -> None:
        self.last_system = None

    def complete(self, prompt, *, system=None, max_tokens=400, temperature=0.0):
        from crisis_triage.bedrock.client import Completion

        self.last_system = system
        return Completion(text="a reply", latency_ms=1.0)


@pytest.mark.parametrize("route", ["calm", "check_in", "crisis"])
def test_the_prompt_no_longer_contains_the_support_line(route):
    """The line must come from code, not from the model's instructions."""
    client = _CapturingClient()
    out = Responder(client).respond(CRISIS_TEXT, route)
    assert "988" not in client.last_system
    assert "911" not in client.last_system
    # ... and the code adds it where the route requires it. CRISIS_TEXT carries no
    # distress phrase, so the calm route correctly omits it.
    if route in {"check_in", "crisis"}:
        assert US_SUPPORT_LINE in out.text
    else:
        assert US_SUPPORT_LINE not in out.text


def test_calm_prompt_forbids_crisis_resources():
    client = _CapturingClient()
    Responder(client).respond(ROUTING_COMPLAINT, "calm")
    system = client.last_system.lower()
    assert "do not mention crisis resources" in system
    assert "hotlines" in system