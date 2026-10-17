"""The state sent to Decider must carry the earlier turns.

This is the property the demo's turn-2/turn-3 scores depend on: Decider sees the last
N user turns, so changing turn 1 must change what turns 2 and 3 are judged against.
If that ever regresses -- for example by sending only the latest message -- the
`escalating` question silently loses its comparison point and every per-turn claim in
the docs becomes wrong, without any error.
"""

from __future__ import annotations

import pytest

from crisis_triage.policies.confidence_band import ConfidenceBandPolicy
from crisis_triage.policies.crisis_policy import CrisisTriagePolicy

TURN_1_A = "My manager piled three deadlines on me today and I am completely worn out."
TURN_1_B = "My cat knocked a plant off the windowsill this morning."
TURN_2 = "I am really struggling and I do not know how much longer I can keep going."
TURN_3 = "I decided tonight is the night. I have the pills ready."


def _policy(fake_client) -> CrisisTriagePolicy:
    return CrisisTriagePolicy(
        ConfidenceBandPolicy(fake_client, low=0.15, high=0.30), None
    )


def _run(policy: CrisisTriagePolicy, turns: list[str]) -> list[str | None]:
    """Feed turns one at a time, collecting the state Decider saw on each."""
    states = []
    for i in range(1, len(turns) + 1):
        decision = policy.band_policy.decide(turns[:i])
        states.append(decision.state)
    return states


@pytest.mark.parametrize("index", [1, 2])
def test_conversations_differing_only_in_turn_1_have_different_states(fake_client, index):
    """Turn 2 and turn 3 must both reflect which turn 1 came first."""
    states_a = _run(_policy(fake_client), [TURN_1_A, TURN_2, TURN_3])
    states_b = _run(_policy(fake_client), [TURN_1_B, TURN_2, TURN_3])

    # Turn 1 is trivially different; the assertion that matters is about later turns.
    assert states_a[index] != states_b[index], (
        f"turn {index + 1} state is identical across the two conversations -- the "
        "earlier turn is not reaching Decider"
    )


def test_later_states_contain_their_own_turn_1(fake_client):
    states_a = _run(_policy(fake_client), [TURN_1_A, TURN_2, TURN_3])
    states_b = _run(_policy(fake_client), [TURN_1_B, TURN_2, TURN_3])

    for state in states_a[1:]:
        assert TURN_1_A in state
        assert TURN_1_B not in state
    for state in states_b[1:]:
        assert TURN_1_B in state
        assert TURN_1_A not in state


def test_state_grows_with_the_conversation(fake_client):
    states = _run(_policy(fake_client), [TURN_1_A, TURN_2, TURN_3])
    assert states[0] == TURN_1_A
    assert states[0] != states[1] != states[2]
    assert TURN_2 in states[1] and TURN_2 in states[2]
    assert TURN_3 in states[2]
    assert len(states[0]) < len(states[1]) < len(states[2])


def test_oldest_turns_are_dropped_past_max_turns(fake_client):
    policy = CrisisTriagePolicy(
        ConfidenceBandPolicy(fake_client, low=0.15, high=0.30, max_turns=2), None
    )
    states = _run(policy, [TURN_1_A, TURN_2, TURN_3])
    assert TURN_1_A not in states[2], "max_turns must bound the state"
    assert TURN_2 in states[2] and TURN_3 in states[2]


def test_decision_records_the_question_set_sent(fake_client):
    decision = _policy(fake_client).band_policy.decide([TURN_1_A])
    assert decision.questions is not None
    assert "is_emergency" in decision.questions
    assert "escalating" in decision.questions
    # The recorded set is the same object the client was asked for.
    assert list(fake_client.calls[-1]["questions"]) == list(decision.questions)


def test_state_is_recorded_even_when_decider_fails(fake_client):
    from crisis_triage.decider_client import DeciderError

    fake_client.error = DeciderError("connection refused")
    decision = _policy(fake_client).band_policy.decide_or_fallback([TURN_1_A, TURN_3])
    assert decision.state is not None, "an outage must still record what we tried to send"
    assert TURN_1_A in decision.state and TURN_3 in decision.state
    assert decision.questions is not None


def test_handler_records_the_same_state_it_sent(fake_client):
    """The Strands path feeds triage_state directly; the recorded state must match."""
    from crisis_triage.policies.crisis_policy import TriageOutcome

    policy = _policy(fake_client)
    outcome = policy.triage([TURN_1_A, TURN_2])
    assert isinstance(outcome, TriageOutcome)
    assert outcome.band.state is not None
    assert TURN_1_A in outcome.band.state and TURN_2 in outcome.band.state
    # What Decider was actually called with is the recorded string.
    assert fake_client.calls[-1]["state"] == outcome.band.state