"""The safety-critical part: routing must never let an emergency fall through.

These tests encode the failure mode that the fixture sweep exposed -- Decider alone
misses emergencies -- so they are the regression guard for the architecture, not just
for the code.
"""

from __future__ import annotations

import pytest

from crisis_triage.decider_client import DeciderError
from crisis_triage.policies.confidence_band import ConfidenceBandPolicy, Route
from crisis_triage.policies.crisis_policy import CrisisTriagePolicy

LOW, HIGH = 0.15, 0.30


def build(fake_client, safety=None, *,
          max_turns: int = 6,
          esc_threshold: float | None = None) -> CrisisTriagePolicy:
    band = ConfidenceBandPolicy(fake_client, low=LOW, high=HIGH, max_turns=max_turns)
    return CrisisTriagePolicy(band, safety, esc_threshold=esc_threshold)


def test_calm_when_every_signal_is_quiet(fake_client, fake_safety):
    fake_client.emergency = 0.05
    fake_client.escalating = 0.40
    fake_safety.emergency = False
    outcome = build(fake_client, fake_safety).triage(["my payouts are failing"])
    assert outcome.route is Route.CALM
    assert outcome.triggered_by == []


def test_high_band_routes_to_crisis(fake_client, fake_safety):
    fake_client.emergency = 0.9
    outcome = build(fake_client, fake_safety).triage(["help"])
    assert outcome.route is Route.CRISIS
    assert "decider_high_band" in outcome.triggered_by


def test_bedrock_safety_alone_rescues_a_missed_emergency(fake_client, fake_safety):
    """The measured failure: plan + means scored 0.21 while a check-in reached 0.18.

    Decider says calm; the safety check says emergency. The OR must route to crisis.
    """
    fake_client.emergency = 0.21
    fake_client.escalating = 0.40
    fake_safety.emergency = True
    outcome = build(fake_client, fake_safety).triage(["I have a plan and I bought what I need"])
    assert outcome.route is Route.CRISIS
    assert outcome.triggered_by == ["bedrock_safety"]


def test_escalating_alone_routes_to_crisis(fake_client, fake_safety):
    fake_client.emergency = 0.10
    fake_client.escalating = 0.80
    fake_safety.emergency = False
    outcome = build(fake_client, fake_safety).triage(["I decided tonight is the night"])
    assert outcome.route is Route.CRISIS
    assert "decider_escalating" in outcome.triggered_by


def test_decider_failure_fails_toward_crisis_not_silence(fake_client, fake_safety):
    """Rule 5: unreachable Decider must never produce a calm reply."""
    fake_client.error = DeciderError("connection refused")
    fake_safety.emergency = False  # even with a clear safety check, fail safe
    outcome = build(fake_client, fake_safety).triage(["anything"])
    assert outcome.route is Route.CRISIS
    assert "decider_unavailable" in outcome.triggered_by
    assert outcome.band.fallback_reason is not None


def test_decider_failure_without_safety_check_still_fails_safe(fake_client):
    fake_client.error = DeciderError("connection refused")
    outcome = build(fake_client, safety=None).triage(["anything"])
    assert outcome.route is Route.CRISIS
    assert "decider_unavailable" in outcome.triggered_by


def test_mid_band_with_quiet_signals_is_check_in(fake_client, fake_safety):
    fake_client.emergency = 0.20
    fake_client.escalating = 0.40
    fake_safety.emergency = False
    outcome = build(fake_client, fake_safety).triage(["I am really struggling lately"])
    assert outcome.route is Route.CHECK_IN
    assert outcome.triggered_by == []


def test_escalation_threshold_is_configurable(fake_client, fake_safety):
    fake_client.emergency = 0.10
    fake_client.escalating = 0.65
    fake_safety.emergency = False
    # 0.65 clears a 0.60 bar but not a 0.70 one, so the two policies disagree.
    assert build(fake_client, fake_safety, esc_threshold=0.60).triage(["x"]).route is Route.CRISIS
    assert build(fake_client, fake_safety, esc_threshold=0.90).triage(["x"]).route is Route.CALM


def test_without_a_safety_checker_decider_alone_governs(fake_client):
    """Documents the unsafe configuration the CLI warns about via --no-safety-check."""
    fake_client.emergency = 0.21  # a real plan + means, scored by Decider as low
    fake_client.escalating = 0.40
    outcome = build(fake_client, safety=None).triage(["I have a plan and I bought what I need"])
    # 0.21 sits in the genuinely ambiguous middle, so Decider alone cannot do better
    # than CHECK_IN here -- it never reaches CRISIS. That is the point: with no safety
    # check, someone with a plan and the means gets a gentle clarifying question
    # instead of crisis resources. Hence the check is on by default.
    assert outcome.route is not Route.CRISIS, (
        "without the Bedrock check this emergency never reaches CRISIS -- this is why "
        "the safety check is on by default and --no-safety-check is not recommended"
    )


def test_guidance_is_route_specific(fake_client, fake_safety):
    policy = build(fake_client, fake_safety)

    fake_safety.emergency = True
    assert "crisis resources" in policy.triage(["x"]).guidance.lower()

    fake_safety.emergency = False
    fake_client.emergency = 0.20  # mid band -> check-in
    assert "one gentle" in policy.triage(["x"]).guidance.lower()

    fake_client.emergency = 0.05  # low band -> calm
    assert "never dismiss" in policy.triage(["x"]).guidance.lower()


def test_explain_is_safe_to_print(fake_client, fake_safety):
    fake_client.emergency = 0.2143
    fake_client.escalating = 0.7239
    fake_safety.emergency = True
    text = build(fake_client, fake_safety).triage(["x"]).explain()
    assert "0.2143" in text and "bedrock_safety=True" in text