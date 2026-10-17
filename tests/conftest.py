"""Shared test doubles.

Nothing here touches the network. CI must pass with no Decider server and no Bedrock
credentials (docs/HACKATHON.md section 7, pitfall 7), so every test either mocks the
client or drives the pure routing logic directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from crisis_triage.decider_client import (  # noqa: E402
    ChoiceAnswer,
    DeciderError,
    DeciderResponse,
    NoulAnswer,
    ScoreAnswer,
)
from crisis_triage.bedrock.safety_check import SafetyVerdict  # noqa: E402


class FakeDeciderClient:
    """Stands in for DeciderClient. Records the requests it was given."""

    def __init__(
        self,
        *,
        emergency: float = 0.05,
        escalating: float = 0.5,
        crisis_type: str = "not_crisis",
        urgency: float = 3.0,
        error: Exception | None = None,
        health_ok: bool = True,
    ) -> None:
        self.emergency = emergency
        self.escalating = escalating
        self.crisis_type = crisis_type
        self.urgency = urgency
        self.error = error
        self.health_ok = health_ok
        self.calls: list[dict] = []

    def health(self, timeout_s: float = 10.0) -> dict:
        if not self.health_ok:
            raise DeciderError("fake: server down")
        return {"status": "ok", "model": "fake-decider", "device": "cuda", "num_slots": 24}

    def is_healthy(self, timeout_s: float = 10.0) -> bool:
        return self.health_ok

    def evaluate(self, state: str, questions: dict, timeout_s=None) -> DeciderResponse:
        self.calls.append({"state": state, "questions": list(questions)})
        if self.error is not None:
            raise self.error
        answers = {}
        for name in questions:
            if name == "is_emergency":
                answers[name] = NoulAnswer(probability=self.emergency)
            elif name == "escalating":
                answers[name] = NoulAnswer(probability=self.escalating)
            elif name == "crisis_type":
                answers[name] = ChoiceAnswer(
                    choice=self.crisis_type,
                    probabilities={self.crisis_type: 0.6, "not_crisis": 0.4},
                    confidence=0.2,
                )
            elif name == "urgency_level":
                answers[name] = ScoreAnswer(
                    score=self.urgency,
                    legend={"0": "immediate", "1": "urgent", "2": "elevated", "3": "routine"},
                    probabilities={"0": 0.05, "1": 0.1, "2": 0.25, "3": 0.6},
                    confidence=0.5,
                )
        return DeciderResponse(
            answers=answers,
            model="fake-decider",
            latency_ms=150.0,
            input_tokens=200,
            output_tokens=4,
        )


class FakeSafetyChecker:
    """Stands in for SafetyChecker."""

    def __init__(self, emergency: bool = False, *, failed: bool = False) -> None:
        self.emergency = emergency
        self.failed = failed
        self.calls: list[str] = []

    def check(self, text: str) -> SafetyVerdict:
        self.calls.append(text)
        return SafetyVerdict(
            emergency=self.emergency,
            reason="fake",
            latency_ms=900.0,
            input_tokens=100,
            output_tokens=20,
            failed=self.failed,
        )


class FakeResponder:
    """Stands in for Responder; echoes the route so tests can assert on it.

    It applies the real `append_support_line` helper, so agent-level tests see the
    same support-line behaviour a real responder produces rather than a stub that
    would silently hide a regression.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def respond(self, user_text: str, route_value: str, safety_note: str = ""):
        from crisis_triage.bedrock.responders import ResponderResult
        from crisis_triage.resources import append_support_line

        self.calls.append((user_text, route_value))
        return ResponderResult(
            text=append_support_line(f"[{route_value}] reply", route_value, user_text),
            latency_ms=800.0,
            input_tokens=120,
            output_tokens=40,
        )


@pytest.fixture
def fake_client() -> FakeDeciderClient:
    return FakeDeciderClient()


@pytest.fixture
def fake_safety() -> FakeSafetyChecker:
    return FakeSafetyChecker()


@pytest.fixture
def fake_responder() -> FakeResponder:
    return FakeResponder()