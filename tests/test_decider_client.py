"""DeciderClient against the *real* response shape, without a server.

The fixtures here are trimmed copies of a genuine `/v1/systemone` response captured
from the live server on 2026-10-17 (docs/GROUND_TRUTH.md section 2), so these tests
pin the contract rather than a guess about it.
"""

from __future__ import annotations

import json

import pytest

from crisis_triage.decider_client import (
    ChoiceAnswer,
    DeciderClient,
    DeciderError,
    NoulAnswer,
    ScoreAnswer,
)

REAL_SHAPE = {
    "model": "strands-decider-2B-hobson-v19",
    "answers": {
        "is_emergency": {"type": "noul", "noul": 0.2158},
        "crisis_type": {
            "type": "choice",
            "choice": "suicide_self_harm",
            "probabilities": {"suicide_self_harm": 0.5329, "not_crisis": 0.4457},
            "confidence": 0.3772,
        },
        "urgency_level": {
            "type": "score",
            "score": 1.7526,
            "legend": {"0": "immediate", "1": "urgent"},
            "probabilities": {"0": 0.1442, "1": 0.2053},
            "confidence": 0.4363,
        },
    },
    "usage": {"input_tokens": 447, "output_tokens": 4},
    "latency_ms": 370.51,
}


@pytest.fixture
def fake_client() -> DeciderClient:
    return DeciderClient("http://127.0.0.1:8000", timeout_s=5)


def test_parses_the_real_response_shape(fake_client, monkeypatch):
    monkeypatch.setattr(fake_client, "_post", lambda *a, **k: REAL_SHAPE)
    response = fake_client.evaluate("state", {"is_emergency": {"type": "noul", "instructions": "?"}})

    assert response.model == "strands-decider-2B-hobson-v19"
    assert response.latency_ms == pytest.approx(370.51)
    assert response.input_tokens == 447 and response.output_tokens == 4
    assert isinstance(response.answers["is_emergency"], NoulAnswer)
    assert isinstance(response.answers["crisis_type"], ChoiceAnswer)
    assert isinstance(response.answers["urgency_level"], ScoreAnswer)
    assert response.noul("is_emergency") == pytest.approx(0.2158)
    assert response.choice("crisis_type").choice == "suicide_self_harm"
    assert response.score("urgency_level").score == pytest.approx(1.7526)
    assert response.score("urgency_level").legend["0"] == "immediate"


def test_missing_answer_raises_rather_than_guessing(fake_client, monkeypatch):
    monkeypatch.setattr(fake_client, "_post", lambda *a, **k: REAL_SHAPE)
    response = fake_client.evaluate("state", {"is_emergency": {"type": "noul", "instructions": "?"}})
    with pytest.raises(DeciderError, match="no answer named"):
        response.noul("escalating")


def test_wrong_answer_type_raises(fake_client, monkeypatch):
    monkeypatch.setattr(fake_client, "_post", lambda *a, **k: REAL_SHAPE)
    response = fake_client.evaluate("state", {"is_emergency": {"type": "noul", "instructions": "?"}})
    with pytest.raises(DeciderError, match="expected ChoiceAnswer"):
        response.choice("is_emergency")


def test_unknown_answer_type_raises(fake_client, monkeypatch):
    payload = {"answers": {"weird": {"type": "telepathy", "value": 1}}, "usage": {}}
    monkeypatch.setattr(fake_client, "_post", lambda *a, **k: payload)
    with pytest.raises(DeciderError, match="unknown type"):
        fake_client.evaluate("state", {"weird": {}})


def test_empty_answers_raises(fake_client, monkeypatch):
    monkeypatch.setattr(fake_client, "_post", lambda *a, **k: {"model": "x", "answers": {}})
    with pytest.raises(DeciderError, match="no usable 'answers'"):
        fake_client.evaluate("state", {"q": {}})


def test_empty_state_is_rejected_before_the_call(fake_client):
    with pytest.raises(DeciderError, match="state must not be empty"):
        fake_client.evaluate("   ", {"q": {}})


def test_no_questions_is_rejected(fake_client):
    with pytest.raises(DeciderError, match="at least one question"):
        fake_client.evaluate("state", {})


def test_latency_is_optional_and_not_fatal(fake_client, monkeypatch):
    payload = {"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {}}
    monkeypatch.setattr(fake_client, "_post", lambda *a, **k: payload)
    response = fake_client.evaluate("state", {"q": {}})
    assert response.latency_ms != response.latency_ms  # NaN, not a crash


def test_defaults_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("DECIDER_URL", "http://127.0.0.1:9999/")
    monkeypatch.setenv("DECIDER_TIMEOUT_S", "12")
    fake_client = DeciderClient()
    assert fake_client.base_url == "http://127.0.0.1:9999"
    assert fake_client.timeout_s == 12.0


def test_connection_failure_becomes_a_decider_error():
    """No server on this port -- must raise DeciderError, never something else."""
    fake_client = DeciderClient("http://127.0.0.1:1", timeout_s=1)
    with pytest.raises(DeciderError):
        fake_client.evaluate("state", {"q": {"type": "noul", "instructions": "?"}})


def test_is_healthy_returns_false_instead_of_raising():
    fake_client = DeciderClient("http://127.0.0.1:1", timeout_s=1)
    assert fake_client.is_healthy() is False