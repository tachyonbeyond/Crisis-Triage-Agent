"""The reusable band pattern itself -- domain-agnostic, no crisis assumptions."""

from __future__ import annotations

import pytest

from crisis_triage.policies.confidence_band import (
    BandPolicyError,
    ConfidenceBandPolicy,
    Route,
)


def make_policy(fake_client, **kwargs) -> ConfidenceBandPolicy:
    kwargs.setdefault("low", 0.15)
    kwargs.setdefault("high", 0.30)
    return ConfidenceBandPolicy(fake_client, **kwargs)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.30, Route.CRISIS),  # band boundaries are inclusive
        (0.99, Route.CRISIS),
        (0.15, Route.CALM),
        (0.01, Route.CALM),
        (0.2999, Route.CHECK_IN),
        (0.1501, Route.CHECK_IN),
    ],
)
def test_bands(fake_client, probability, expected):
    assert make_policy(fake_client).route_for(probability) is expected


def test_decide_uses_the_named_question(fake_client):
    fake_client.emergency = 0.42
    decision = make_policy(fake_client).decide(["hello"])
    assert decision.route is Route.CRISIS
    assert decision.probability == pytest.approx(0.42)
    assert decision.question == "is_emergency"


def test_decide_batches_every_question_in_one_call(fake_client):
    make_policy(fake_client).decide(["hello"])
    assert len(fake_client.calls) == 1, "questions must share one request to reuse the KV prefix"


def test_state_keeps_only_the_last_n_user_turns(fake_client):
    policy = make_policy(fake_client, max_turns=2)
    policy.decide(["one", "two", "three", "four"])
    state = fake_client.calls[0]["state"]
    assert "three" in state and "four" in state
    assert "one" not in state and "two" not in state


def test_escalation_is_reported_not_acted_on(fake_client):
    """The core pattern is a pure band rule; escalation is surfaced for the caller."""
    fake_client.emergency = 0.05
    fake_client.escalating = 0.95
    decision = make_policy(fake_client).decide(["hello"])
    assert decision.route is Route.CALM
    assert decision.escalating_probability == pytest.approx(0.95)


def test_low_above_high_is_rejected(fake_client):
    with pytest.raises(BandPolicyError):
        ConfidenceBandPolicy(fake_client, low=0.8, high=0.2)


def test_missing_thresholds_are_rejected_rather_than_defaulted(fake_client, monkeypatch):
    monkeypatch.delenv("CRISIS_LOW", raising=False)
    monkeypatch.delenv("CRISIS_HIGH", raising=False)
    with pytest.raises(BandPolicyError, match="thresholds not set"):
        ConfidenceBandPolicy(fake_client)


def test_thresholds_can_come_from_the_environment(fake_client, monkeypatch):
    monkeypatch.setenv("CRISIS_LOW", "0.2")
    monkeypatch.setenv("CRISIS_HIGH", "0.4")
    policy = ConfidenceBandPolicy(fake_client)
    assert (policy.low, policy.high) == (0.2, 0.4)


def test_unknown_question_is_rejected(fake_client):
    with pytest.raises(BandPolicyError):
        make_policy(fake_client, question="not_a_question")


def test_empty_state_is_rejected(fake_client):
    with pytest.raises(ValueError):
        make_policy(fake_client).decide([])
    with pytest.raises(ValueError):
        make_policy(fake_client).decide(["   "])