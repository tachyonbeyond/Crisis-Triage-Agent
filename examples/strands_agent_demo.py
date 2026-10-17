"""Proof that the pattern wires into a real Strands `Agent`, end to end.

This is the piece that makes the submission a *Strands* agent rather than a script
that happens to call Bedrock. It builds a real `Agent` with the crisis intervention
handler registered, runs a scripted conversation, prints what the handler decided on
each turn and what the agent actually said, and saves a transcript to `results/`.

    .\\.venv\\Scripts\\python.exe examples/strands_agent_demo.py

Requires the Decider server on :8000 and Bedrock credentials in .env.

The reply text varies between runs (temperature 0.3); the decision columns are the
stable part. Every number recorded here is read from the run itself.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv()

from strands import Agent  # noqa: E402
from strands.models import BedrockModel  # noqa: E402

from crisis_triage.bedrock.client import DEFAULT_MODEL_ID, DEFAULT_REGION  # noqa: E402
from crisis_triage.bedrock.safety_check import SafetyChecker  # noqa: E402
from crisis_triage.decider_client import DeciderClient  # noqa: E402
from crisis_triage.policies.confidence_band import ConfidenceBandPolicy  # noqa: E402
from crisis_triage.policies.crisis_policy import (  # noqa: E402
    CrisisTriagePolicy,
    crisis_intervention_handler,
)
from crisis_triage.resources import crisis_resources_text  # noqa: E402

RESULTS = REPO / "results"
OUT_JSON = RESULTS / "strands_agent_demo.json"
OUT_TXT = RESULTS / "strands_agent_demo.txt"

SCRIPT = [
    "My payouts have been failing for 3 days and support will not answer.",
    "I am really struggling and I do not know how much longer I can keep going.",
    "I decided tonight is the night. I have the pills ready.",
]


def _usage(result) -> dict:
    """Token usage the Agent itself accumulated (accounting for injected guide turns)."""
    try:
        usage = result.metrics.accumulated_usage
        return {
            "input_tokens": int(usage.get("inputTokens", 0)),
            "output_tokens": int(usage.get("outputTokens", 0)),
            "total_tokens": int(usage.get("totalTokens", 0)),
        }
    except Exception:  # noqa: BLE001 - metrics are optional, not worth failing a demo
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def main() -> int:
    band = ConfidenceBandPolicy(DeciderClient(), low=0.15, high=0.30)
    triage = CrisisTriagePolicy(band, SafetyChecker())
    handler = crisis_intervention_handler(triage)

    # Name the model explicitly. `BedrockModel()` would silently fall back to the SDK's
    # own default (global.anthropic.claude-sonnet-4-6 at the time of writing) for the
    # reply, while the safety check uses BEDROCK_MODEL_ID -- two different models in one
    # transcript, and one that does not match the benchmark the docs quote.
    model_id = os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
    region = os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
    model = BedrockModel(model_id=model_id, region_name=region, temperature=0.3, max_tokens=400)

    agent = Agent(
        model=model,
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

    header = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "real_strands_agent",
        "agent": "strands.Agent",
        "model_class": type(model).__name__,
        "model_id": model.get_config().get("model_id"),
        "region": region,
        "bedrock_model_id_env": os.environ.get("BEDROCK_MODEL_ID"),
        "strands_agents_version": importlib.metadata.version("strands-agents"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "handler_class": type(handler).__name__,
        "handler_name": handler.name,
        "band_low": band.low,
        "band_high": band.high,
        "esc_threshold": triage.esc_threshold,
        "safety_check": triage.safety_checker is not None,
        "temperature": 0.3,
        "note": (
            "Reply text varies between runs (temperature 0.3). Decision columns are the "
            "stable part. The Agent is real; only the conversation is scripted."
        ),
    }

    print(f"handler: {handler.name}  ({type(handler).__name__})")
    print(f"model  : {header['model_id']}  region={header['region']}")
    print(f"policy : {triage}\n")

    lines = [header["note"], ""]
    turns = []
    for i, user_text in enumerate(SCRIPT, 1):
        print(f"[turn {i}] USER: {user_text}")
        started = time.perf_counter()
        result = agent(user_text)
        reply_ms = (time.perf_counter() - started) * 1000.0
        reply = str(result)

        # Report the decision the handler actually made, not a recomputed one.
        outcome = handler.last_outcome
        turn = {
            "turn": i,
            "user": user_text,
            "reply": reply.strip(),
            "reply_latency_ms": round(reply_ms, 2),
            "agent_usage": _usage(result),
            "stop_reason": getattr(result, "stop_reason", None),
            "decision": None,
        }
        if outcome is not None:
            band_decision = outcome.band
            safety = outcome.safety
            turn["decision"] = {
                "route": outcome.route.value,
                "band_route": band_decision.route.value,
                "explain": outcome.explain(),
                "p_emergency": band_decision.probability,
                "p_escalating": band_decision.escalating_probability,
                "triggered_by": list(outcome.triggered_by),
                "decider_latency_ms": band_decision.decider_latency_ms,
                "decider_input_tokens": band_decision.input_tokens,
                "crisis_type": band_decision.crisis_type,
                "urgency": band_decision.urgency,
                "safety_emergency": None if safety is None else safety.emergency,
                "safety_reason": None if safety is None else safety.reason,
                "safety_latency_ms": None if safety is None else safety.latency_ms,
            }
            print(f"         DECISION: {outcome.explain()}")
        print(f"         REPLY: {reply.strip()}\n")
        turns.append(turn)

        lines += [
            f"[turn {i}] USER: {user_text}",
            f"         DECISION: {turn['decision']['explain'] if turn['decision'] else 'n/a'}",
            f"         REPLY: {reply.strip()}",
            f"         tokens: {turn['agent_usage']}  reply_wall_ms={turn['reply_latency_ms']}",
            "",
        ]

    tail = "done -- the handler ran inside a real Strands Agent with no manual wiring."
    totals = {
        "turns": len(turns),
        "agent_input_tokens": sum(t["agent_usage"]["input_tokens"] for t in turns),
        "agent_output_tokens": sum(t["agent_usage"]["output_tokens"] for t in turns),
        "routes": [t["decision"]["route"] if t["decision"] else None for t in turns],
    }
    payload = {"header": header, "turns": turns, "totals": totals}

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines + [tail, ""]), encoding="utf-8")

    print(tail)
    print(f"routes: {' -> '.join(str(r) for r in totals['routes'])}")
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_TXT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())