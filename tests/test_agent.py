"""The orchestrator end to end, with both backends faked.

This is the mock-based path that must pass with no Decider server and no Bedrock
credentials (docs/HACKATHON.md section 7, pitfall 7).
"""

from __future__ import annotations

import pytest

from crisis_triage.bedrock.responders import ResponderResult
from crisis_triage.decider_client import DeciderError
from crisis_triage.main import CrisisTriageAgent
from crisis_triage.policies.confidence_band import ConfidenceBandPolicy, Route


def build_agent(fake_client, safety, responder) -> CrisisTriageAgent:
    band = ConfidenceBandPolicy(fake_client, low=0.15, high=0.30)
    return CrisisTriageAgent(
        band_policy=band,
        safety_checker=safety,
        responder=responder,
        use_safety_check=True,
    )


def test_quiet_turn_stays_calm(fake_client, fake_safety, fake_responder):
    fake_client.emergency = 0.05
    result = build_agent(fake_client, fake_safety, fake_responder).turn("my payouts are failing")
    assert result.route is Route.CALM
    assert result.reply == "[calm] reply"
    assert fake_responder.calls[-1][1] == "calm"


def test_bedrock_safety_forces_a_crisis_reply(fake_client, fake_safety, fake_responder):
    fake_client.emergency = 0.20  # below the high band -- Decider alone would say calm
    fake_safety.emergency = True
    result = build_agent(fake_client, fake_safety, fake_responder).turn("I have the pills ready")
    assert result.route is Route.CRISIS
    assert result.reply == "[crisis] reply"
    assert fake_responder.calls[-1][1] == "crisis"


def test_decider_outage_still_produces_a_crisis_reply(fake_client, fake_safety, fake_responder):
    fake_client.error = DeciderError("connection refused")
    fake_safety.emergency = False
    result = build_agent(fake_client, fake_safety, fake_responder).turn("hello")
    assert result.route is Route.CRISIS, "a triage outage must not route to calm"
    assert "decider_unavailable" in result.outcome.triggered_by


def test_conversation_history_feeds_the_state(fake_client, fake_safety, fake_responder):
    agent = build_agent(fake_client, fake_safety, fake_responder)
    agent.turn("first message")
    agent.turn("second message")
    assert "first message" in fake_client.calls[-1]["state"]
    assert "second message" in fake_client.calls[-1]["state"]


def test_reset_clears_history(fake_client, fake_safety, fake_responder):
    agent = build_agent(fake_client, fake_safety, fake_responder)
    agent.turn("first message")
    agent.reset()
    agent.turn("only message")
    assert "first message" not in fake_client.calls[-1]["state"]


def test_history_records_the_route_each_turn(fake_client, fake_safety, fake_responder):
    fake_client.emergency = 0.05
    agent = build_agent(fake_client, fake_safety, fake_responder)
    agent.turn("a")
    fake_safety.emergency = True
    agent.turn("b")
    assert [turn.route for turn in agent.history] == [Route.CALM, Route.CRISIS]


def test_a_bedrock_reply_failure_falls_back_visibly(fake_client, fake_safety, monkeypatch):
    class FailingResponder:
        def respond(self, user_text, route_value, safety_note=""):
            return ResponderResult(
                text="fallback text",
                latency_ms=float("nan"),
                used_fallback=True,
                error="BedrockError: throttled",
            )

    fake_client.emergency = 0.05
    agent = build_agent(fake_client, fake_safety, FailingResponder())
    result = agent.turn("hello")
    assert result.used_fallback is True
    assert result.error is not None


def test_health_reports_both_backends_without_raising(fake_client, fake_safety, fake_responder):
    agent = build_agent(fake_client, fake_safety, fake_responder)
    report = agent.health()
    assert report["decider"]["ok"] is True
    assert report["thresholds"] == {"low": 0.15, "high": 0.30, "escalating": 0.70}
    # The fake responder has no Bedrock client, so the report must say so rather
    # than crash -- a diagnostic that can throw is not a diagnostic.
    assert report["bedrock"]["ok"] is False
    assert "error" in report["bedrock"]


def test_health_reports_a_down_decider_instead_of_raising(fake_client, fake_safety, fake_responder):
    fake_client.health_ok = False
    agent = build_agent(fake_client, fake_safety, fake_responder)
    assert agent.health()["decider"]["ok"] is False