"""The Bedrock safety check, including the rule that it must fail toward caution."""

from __future__ import annotations

import pytest

from crisis_triage.bedrock.client import BedrockError, Completion, extract_json_object
from crisis_triage.bedrock.safety_check import SafetyChecker


class StubClient:
    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error

    def complete_json(self, prompt, *, system, max_tokens=200):
        if self.error is not None:
            raise self.error
        return extract_json_object(self.reply or ""), Completion(
            text=self.reply or "", latency_ms=900.0, input_tokens=100, output_tokens=15
        )


def test_clear_emergency():
    verdict = SafetyChecker(StubClient('{"emergency": true, "reason": "active plan"}')).check("x")
    assert verdict.emergency is True and not verdict.failed
    assert verdict.reason == "active plan"


def test_clear_non_emergency():
    verdict = SafetyChecker(StubClient('{"emergency": false, "reason": "venting"}')).check("x")
    assert verdict.emergency is False and not verdict.failed


def test_verdict_wrapped_in_prose_is_still_parsed():
    reply = 'Here is my answer:\n```json\n{"emergency": true, "reason": "overdose"}\n```\nHope that helps.'
    assert SafetyChecker(StubClient(reply)).check("x").emergency is True


def test_bedrock_failure_fails_closed_by_default(monkeypatch):
    monkeypatch.delenv("SAFETY_FAIL_CLOSED", raising=False)
    verdict = SafetyChecker(StubClient(error=BedrockError("throttled"))).check("x")
    assert verdict.emergency is True, "an unavailable safety check must not read as safe"
    assert verdict.failed is True


def test_bedrock_failure_can_be_opted_out_of(monkeypatch):
    monkeypatch.setenv("SAFETY_FAIL_CLOSED", "0")
    verdict = SafetyChecker(StubClient(error=BedrockError("throttled"))).check("x")
    assert verdict.emergency is False and verdict.failed is True


def test_unparseable_verdict_fails_closed(monkeypatch):
    monkeypatch.delenv("SAFETY_FAIL_CLOSED", raising=False)
    verdict = SafetyChecker(StubClient('{"emergency": "maybe"}')).check("x")
    assert verdict.emergency is True and verdict.failed is True


@pytest.mark.parametrize("text", ["", "no json here", "{not json}", '["a list"]'])
def test_extract_json_object_rejects_junk(text):
    with pytest.raises(BedrockError):
        extract_json_object(text)


def test_extract_json_object_takes_the_first_object():
    assert extract_json_object('prefix {"a": 1} suffix')["a"] == 1