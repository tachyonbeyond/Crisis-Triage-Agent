"""Integration with the **real** `strands.interventions` classes.

The rest of the suite uses test doubles for the policy inputs. This file deliberately
does not: it imports the installed SDK and drives the real `InterventionRegistry`,
`HookRegistry` and `Agent`, so the wiring our handler relies on is exercised rather
than assumed. Every assertion here is offline -- no Decider server and no AWS
credentials -- because the model and the Decider client are faked *below* the
intervention layer, while the intervention layer itself is the real thing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# Real SDK imports, at module level, on purpose.
from strands.agent.agent import Agent
from strands.hooks.events import BeforeInvocationEvent, BeforeModelCallEvent
from strands.hooks.registry import HookRegistry
from strands.interventions import Guide, InterventionHandler, Proceed
from strands.interventions.registry import InterventionRegistry
from strands.models.model import Model

from crisis_triage.policies.confidence_band import ConfidenceBandPolicy, Route
from crisis_triage.policies.crisis_policy import (
    CrisisTriagePolicy,
    TriageOutcome,
    crisis_intervention_handler,
)

LOW, HIGH = 0.15, 0.30


def _agent_stub(messages: list[dict] | None = None) -> Any:
    """The `agent` attribute real events carry. Only `.messages` is touched."""
    return type("AgentStub", (), {"messages": messages if messages is not None else []})()


def _event(event_type, **kwargs):
    return event_type(agent=_agent_stub(), **kwargs)


def _invocation_turn(hook_registry: HookRegistry, messages: list[dict]):
    """Run one turn's hook sequence the way the real Agent loop does.

    The loop threads ONE `invocation_state` dict from `BeforeInvocationEvent`
    (agent.py:1522) into every `BeforeModelCallEvent` (event_loop.py:513). Reusing a
    single dict is what lets a handler park state at invocation time and read it at
    model-call time -- so the test must thread it too, not construct a fresh one.
    """
    state: dict[str, Any] = {}
    agent = _agent_stub()
    _dispatch(hook_registry, BeforeInvocationEvent(agent=agent, messages=messages, invocation_state=state))
    return agent, state


def _model_call(hook_registry: HookRegistry, agent, state: dict):
    return _dispatch(hook_registry, BeforeModelCallEvent(agent=agent, invocation_state=state))


def _dispatch(hook_registry: HookRegistry, event):
    """Drive the real hook registry the way an Agent does (async path)."""
    return asyncio.run(hook_registry.invoke_callbacks_async(event))


class FakeModel(Model):
    """Minimal stand-in for a model provider -- no network, no credentials.

    Returns a fixed text reply so a full agent turn can run offline.
    """

    def __init__(self, reply: str = "fake reply") -> None:
        self.reply = reply
        self.seen: list[list[dict]] = []

    def update_config(self, **model_config: Any) -> None:
        return None

    def get_config(self) -> Any:
        return {"model_id": "fake-model", "context_window_limit": 4096}

    def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        # Record what the agent actually sent, so a test can see the injected guide.
        self.seen.append([dict(m) for m in messages])
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": self.reply}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def _policy(fake_client, safety) -> CrisisTriagePolicy:
    return CrisisTriagePolicy(ConfidenceBandPolicy(fake_client, low=LOW, high=HIGH), safety)


# -- 1. the handler is the real base class ---------------------------------


def test_handler_is_a_real_intervention_handler(fake_client, fake_safety):
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))
    assert isinstance(handler, InterventionHandler)
    assert handler.name == "crisis_triage"
    # Only class-level overrides are detected by the SDK; confirm ours are.
    assert type(handler).before_invocation is not InterventionHandler.before_invocation
    assert type(handler).before_model_call is not InterventionHandler.before_model_call


# -- 2. the real registry runs triage on before_invocation -----------------


def test_real_registry_parks_a_triage_outcome(fake_client, fake_safety):
    fake_client.emergency = 0.20  # mid band
    fake_safety.emergency = False
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))

    hooks = HookRegistry()
    InterventionRegistry([handler], hooks)  # registers real callbacks

    messages = [
        {"role": "user", "content": [{"text": "first"}]},
        {"role": "assistant", "content": [{"text": "ok"}]},
        {"role": "user", "content": [{"text": "I am really struggling"}]},
    ]
    event = _event(BeforeInvocationEvent, messages=messages, invocation_state={})
    _dispatch(hooks, event)

    outcome = event.invocation_state["crisis_triage"]
    assert isinstance(outcome, TriageOutcome)
    assert outcome.route is Route.CHECK_IN
    assert handler.last_outcome is outcome
    # The state Decider saw contains only the user turns, newest last.
    assert "first" in fake_client.calls[0]["state"]
    assert "assistant" not in fake_client.calls[0]["state"]


def test_real_registry_fails_safe_when_decider_is_down(fake_client, fake_safety):
    from crisis_triage.decider_client import DeciderError

    fake_client.error = DeciderError("connection refused")
    fake_safety.emergency = False
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))

    hooks = HookRegistry()
    InterventionRegistry([handler], hooks)
    event = _event(
        BeforeInvocationEvent,
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        invocation_state={},
    )
    _dispatch(hooks, event)

    assert handler.last_outcome.route is Route.CRISIS


# -- 3. the real registry injects Guide(feedback) exactly once --------------


def test_guide_feedback_is_injected_once(fake_client, fake_safety):
    fake_client.emergency = 0.31  # high band -> crisis directive
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))

    hooks = HookRegistry()
    InterventionRegistry([handler], hooks)
    agent, state = _invocation_turn(
        hooks, [{"role": "user", "content": [{"text": "I have a plan"}]}]
    )

    # First model call: the real registry appends the guidance as a user message.
    _model_call(hooks, agent, state)
    injected = [m for m in agent.messages if m.get("role") == "user"]
    assert len(injected) == 1
    text = injected[0]["content"][0]["text"]
    assert "SAFETY ROUTING" in text
    assert "crisis" in text.lower()

    # Second model call in the same turn: the outcome was consumed, so nothing new.
    _model_call(hooks, agent, state)
    assert len([m for m in agent.messages if m.get("role") == "user"]) == 1


def test_guide_is_a_real_guide_action(fake_client, fake_safety):
    """Sanity: the handler's return type is the SDK's Guide, not a look-alike."""
    fake_client.emergency = 0.31
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))
    hooks = HookRegistry()
    InterventionRegistry([handler], hooks)
    agent, state = _invocation_turn(hooks, [{"role": "user", "content": [{"text": "x"}]}])
    action = handler.before_model_call(BeforeModelCallEvent(agent=agent, invocation_state=state))
    assert isinstance(action, Guide)
    assert "SAFETY ROUTING" in action.feedback
    # Consumed: a second call is a plain Proceed.
    assert isinstance(
        handler.before_model_call(BeforeModelCallEvent(agent=agent, invocation_state=state)), Proceed
    )


# -- 4. a real Agent accepts the handler and runs a turn offline -----------


def test_real_agent_accepts_the_handler(fake_client, fake_safety):
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))
    agent = Agent(model=FakeModel(), interventions=[handler], callback_handler=None)
    assert handler in agent._intervention_registry.handlers


def test_real_agent_turn_delivers_the_guidance_to_the_model(fake_client, fake_safety):
    """The end-to-end path, with the real Agent loop and a fake model underneath."""
    fake_client.emergency = 0.35  # crisis route via the high band
    fake_safety.emergency = True
    model = FakeModel()
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))
    agent = Agent(model=model, interventions=[handler], callback_handler=None)

    result = agent("I decided tonight is the night. I have the pills ready.")

    # The Agent's string rendering appends a newline; compare the content.
    assert str(result).strip() == "fake reply"
    assert handler.last_outcome.route is Route.CRISIS
    # What the model was actually sent includes the injected routing directive.
    sent = model.seen[-1]
    joined = " ".join(
        block.get("text", "")
        for message in sent
        for block in message.get("content", [])
        if isinstance(block, dict)
    )
    assert "SAFETY ROUTING" in joined


def test_constructing_an_agent_makes_no_network_call(fake_client, fake_safety):
    """The offline guarantee: no Decider call and no Bedrock call at construction."""
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))
    Agent(model=FakeModel(), interventions=[handler], callback_handler=None)
    assert fake_client.calls == []
    assert fake_safety.calls == []


@pytest.mark.parametrize("route_kind", ["calm", "check_in", "crisis"])
def test_guidance_differs_by_route(fake_client, fake_safety, route_kind):
    fake_client.emergency = {"calm": 0.05, "check_in": 0.20, "crisis": 0.40}[route_kind]
    fake_safety.emergency = route_kind == "crisis"
    handler = crisis_intervention_handler(_policy(fake_client, fake_safety))
    hooks = HookRegistry()
    InterventionRegistry([handler], hooks)
    agent, state = _invocation_turn(hooks, [{"role": "user", "content": [{"text": "x"}]}])
    action = handler.before_model_call(BeforeModelCallEvent(agent=agent, invocation_state=state))
    assert isinstance(action, Guide)
    assert "SAFETY ROUTING" in action.feedback
    assert handler.last_outcome.route.value == route_kind