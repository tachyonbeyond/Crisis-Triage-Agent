"""Proof that the pattern wires into a real Strands `Agent`, end to end.

This is the piece that makes the submission a *Strands* agent rather than a script
that happens to call Bedrock. It builds a real `Agent` with the crisis intervention
handler registered, runs a scripted conversation, and prints what the handler decided
on each turn and what the agent actually said.

    .\\.venv\\Scripts\\python.exe examples/strands_agent_demo.py

Requires the Decider server on :8000 and Bedrock credentials in .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from strands import Agent  # noqa: E402
from strands.models import BedrockModel  # noqa: E402

from crisis_triage.bedrock.safety_check import SafetyChecker  # noqa: E402
from crisis_triage.decider_client import DeciderClient  # noqa: E402
from crisis_triage.policies.confidence_band import ConfidenceBandPolicy  # noqa: E402
from crisis_triage.policies.crisis_policy import (  # noqa: E402
    CrisisTriagePolicy,
    crisis_intervention_handler,
)
from crisis_triage.resources import crisis_resources_text  # noqa: E402

SCRIPT = [
    "My payouts have been failing for 3 days and support will not answer.",
    "I am really struggling and I do not know how much longer I can keep going.",
    "I decided tonight is the night. I have the pills ready.",
]


def main() -> int:
    band = ConfidenceBandPolicy(DeciderClient(), low=0.15, high=0.30)
    triage = CrisisTriagePolicy(band, SafetyChecker())
    handler = crisis_intervention_handler(triage)

    agent = Agent(
        model=BedrockModel(temperature=0.3, max_tokens=400),
        system_prompt=(
            "You are a warm, plain-spoken crisis-support assistant in a triage demo, "
            "not a substitute for emergency or professional services. Never dismiss or "
            "minimise what the person said. Never claim to be human. Keep replies to "
            "2-5 sentences. Never state a phone number other than the one given below.\n\n"
            f"Support line to include when asked: {crisis_resources_text()}"
        ),
        interventions=[handler],
        # Silence Strands' default per-token console callback so the handler's own
        # decision line stays readable in the demo.
        callback_handler=None,
    )

    print(f"handler: {handler.name}  ({type(handler).__name__})")
    print(f"policy : {triage}\n")

    for i, user_text in enumerate(SCRIPT, 1):
        print(f"[turn {i}] USER: {user_text}")
        reply = str(agent(user_text))
        # Report the decision the handler actually made, not a recomputed one.
        outcome = handler.last_outcome
        if outcome is not None:
            print(f"         DECISION: {outcome.explain()}")
        print(f"         REPLY: {reply.strip()}\n")

    print("done -- the handler ran inside a real Strands Agent with no manual wiring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())