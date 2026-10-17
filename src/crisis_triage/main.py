"""Crisis Triage Agent -- orchestrator and CLI.

    user turn -> Decider 2B (:8000, local, GPU) -> band + escalation
              -> Bedrock safety check
              -> route: crisis | check_in | calm
              -> Bedrock responder -> reply

The routing logic lives in `policies/`; this module wires it up, keeps the
conversation state, and prints what happened. It deliberately does not re-implement
the policy (docs/HACKATHON.md section 7, pitfall 4).

    python -m crisis_triage.main --health
    python -m crisis_triage.main --test
    python -m crisis_triage.main --demo
    python -m crisis_triage.main --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .bedrock.client import BedrockError, BedrockTextClient
from .bedrock.responders import Responder
from .bedrock.safety_check import SafetyChecker
from .decider_client import DeciderClient, DeciderError
from .policies.confidence_band import BandPolicyError, ConfidenceBandPolicy, Route
from .policies.crisis_policy import CrisisTriagePolicy, TriageOutcome
from .utils.logging import setup_logging

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "test_cases.json"


@dataclass
class TriageResult:
    """One turn's outcome: the routing decision plus the reply the user sees."""

    user_text: str
    outcome: TriageOutcome
    reply: str
    reply_latency_ms: float = float("nan")
    used_fallback: bool = False
    error: str | None = None

    @property
    def route(self) -> Route:
        return self.outcome.route


@dataclass
class Turn:
    user: str
    route: Route
    reply: str


class CrisisTriageAgent:
    """Multi-turn triage agent. Decider decides tone; Bedrock decides safety."""

    def __init__(
        self,
        *,
        band_policy: ConfidenceBandPolicy | None = None,
        safety_checker: SafetyChecker | None = None,
        responder: Responder | None = None,
        use_safety_check: bool = True,
        max_turns: int = 6,
    ) -> None:
        load_dotenv()

        self.decider = band_policy.client if band_policy else DeciderClient()
        self.band_policy = band_policy or ConfidenceBandPolicy(self.decider, max_turns=max_turns)
        self.safety_checker = safety_checker if safety_checker is not None else (
            SafetyChecker() if use_safety_check else None
        )
        self.responder = responder or Responder(BedrockTextClient())
        self.policy = CrisisTriagePolicy(self.band_policy, self.safety_checker)
        self.history: list[Turn] = []
        self.log = setup_logging()

    # -- one turn ----------------------------------------------------------

    def turn(self, user_text: str) -> TriageResult:
        """Triage a new user message against the conversation so far."""
        self.history.append(Turn(user=user_text, route=Route.CALM, reply=""))
        self.history[-1] = Turn(user=user_text, route=Route.CALM, reply="")

        turns = [t.user for t in self.history]
        outcome = self.policy.triage(turns)
        self.history[-1].route = outcome.route

        result = self.responder.respond(
            user_text,
            outcome.route.value,
            safety_note=(outcome.safety.reason if outcome.safety else ""),
        )
        self.history[-1].reply = result.text

        self.log.debug("triage %s", outcome.explain())
        return TriageResult(
            user_text=user_text,
            outcome=outcome,
            reply=result.text,
            reply_latency_ms=result.latency_ms,
            used_fallback=result.used_fallback,
            error=result.error,
        )

    def reset(self) -> None:
        self.history.clear()

    # -- diagnostics -------------------------------------------------------

    def health(self, timeout_s: float = 10.0) -> dict:
        """Report both backends. Never raises -- a down backend is the answer."""
        report: dict = {"decider": {}, "bedrock": {}}

        try:
            info = self.decider.health(timeout_s=timeout_s)
            report["decider"] = {"ok": True, **info}
        except DeciderError as exc:
            report["decider"] = {"ok": False, "error": str(exc)}

        try:
            client = self.responder.client
            completion = client.complete(
                "Reply with exactly: OK", system="Follow the instruction exactly.", max_tokens=8
            )
            report["bedrock"] = {
                "ok": completion.text.strip().startswith("OK"),
                "model_id": client.model_id,
                "region": client.region,
                "reply": completion.text.strip()[:40],
                "latency_ms": round(completion.latency_ms, 1),
            }
        except (BedrockError, AttributeError) as exc:
            # AttributeError covers a responder that has no Bedrock client at all
            # (e.g. a test double). A diagnostic must report, never throw.
            report["bedrock"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        report["thresholds"] = {
            "low": self.band_policy.low,
            "high": self.band_policy.high,
            "escalating": self.policy.esc_threshold,
        }
        report["safety_check"] = self.safety_checker is not None
        return report


def _build_agent(args) -> CrisisTriageAgent:
    # Load .env BEFORE constructing any client or policy: the band thresholds and the
    # Bedrock key both come from the environment, and boto3 does not read .env itself.
    load_dotenv()
    low = args.low if args.low is not None else None
    high = args.high if args.high is not None else None
    band = ConfidenceBandPolicy(DeciderClient(), low=low, high=high, max_turns=args.max_turns)
    safety = None
    if not args.no_safety_check:
        safety = SafetyChecker()
    return CrisisTriageAgent(
        band_policy=band,
        safety_checker=safety,
        responder=Responder(BedrockTextClient()),
        use_safety_check=not args.no_safety_check,
        max_turns=args.max_turns,
    )


# -- commands ---------------------------------------------------------------


def cmd_health(args) -> int:
    """`--health`: probe both backends through our own clients."""
    try:
        agent = _build_agent(args)
    except BandPolicyError as exc:
        print(f"configuration error: {exc}")
        return 2
    report = agent.health()
    print(json.dumps(report, indent=2))
    return 0


def cmd_test(args) -> int:
    """`--test`: run the fixture set and report routing against intended labels."""
    agent = _build_agent(args)
    if not FIXTURES.exists():
        print(f"fixtures not found: {FIXTURES}")
        return 2
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]

    rows = []
    for case in cases:
        agent.reset()
        result = None
        for turn_text in case["turns"]:
            result = agent.turn(turn_text)
        expected = case["label"]
        actual = result.route.value
        agreed = expected == actual or (expected == "check_in" and actual in {"check_in", "crisis"})
        rows.append(
            {
                "id": case["id"],
                "expected": expected,
                "actual": actual,
                "agreed": agreed,
                "p": result.outcome.band.probability,
                "escalating": result.outcome.band.escalating_probability,
                "triggered_by": result.outcome.triggered_by,
            }
        )
        flag = "ok " if agreed else "MISMATCH"
        print(
            f"  {flag} {case['id']:<28} expected={expected:<9} actual={actual:<9} "
            f"p={rows[-1]['p']:.4f} by={','.join(rows[-1]['triggered_by']) or '-'}"
        )

    n_ok = sum(1 for r in rows if r["agreed"])
    crisis = [r for r in rows if r["expected"] == "crisis"]
    crisis_ok = sum(1 for r in crisis if r["actual"] == "crisis")
    # The failure that actually matters: an emergency routed anywhere below CRISIS.
    dangerous = [r for r in crisis if r["actual"] != "crisis"]

    print(f"\nagreement: {n_ok}/{len(rows)}")
    print(f"crisis recall: {crisis_ok}/{len(crisis)}  (a miss here is the dangerous failure)")
    if dangerous:
        print("DANGEROUS MISSES (emergency routed below crisis):")
        for row in dangerous:
            print(f"  {row['id']} -> {row['actual']}  by={','.join(row['triggered_by']) or '-'}")
    else:
        print("no emergency was routed below crisis")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    # Exit non-zero only on the dangerous failure, not on benign band disagreement:
    # Decider's 3-way bands cannot separate cleanly (docs/GROUND_TRUTH.md section 4).
    return 0 if not dangerous else 1


DEMO_SCRIPT = [
    "My payouts have been failing for 3 days and support will not answer.",
    "I have a big exam on Friday and I am really anxious about it.",
    "I am really struggling and I do not know how much longer I can keep going.",
    "I lost my job last week and I feel awful about it.",
    "I do not really see the point in anything anymore.",
    "I decided tonight is the night. I have the pills ready.",
]


def cmd_demo(args) -> int:
    """`--demo`: a scripted multi-turn conversation, printing the decision each turn."""
    agent = _build_agent(args)
    traces = []
    print("=== scripted demo (Decider decides tone, Bedrock decides safety) ===\n")
    for i, user_text in enumerate(DEMO_SCRIPT, 1):
        result = agent.turn(user_text)
        print(f"[turn {i}] USER: {user_text}")
        print(f"         {result.outcome.explain()}")
        print(f"         SUPPORT: {result.reply.strip()}")
        if result.used_fallback:
            print(f"         !! reply fell back to a template: {result.error}")
        print()
        traces.append(
            {
                "turn": i,
                "user": user_text,
                "route": result.route.value,
                "probability": result.outcome.band.probability,
                "escalating_probability": result.outcome.band.escalating_probability,
                "triggered_by": result.outcome.triggered_by,
                "safety_emergency": (
                    result.outcome.safety.emergency if result.outcome.safety else None
                ),
                "decider_latency_ms": result.outcome.band.decider_latency_ms,
                "reply_latency_ms": result.reply_latency_ms,
                "reply": result.reply,
                "used_fallback": result.used_fallback,
            }
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def cmd_interactive(args) -> int:
    """`--interactive`: a real conversation, with the decision printed each turn."""
    agent = _build_agent(args)
    print("Crisis Triage Agent -- interactive. Type 'quit' to exit.")
    print("This is a triage demo, not a substitute for emergency services.\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break
        try:
            result = agent.turn(user_text)
        except (DeciderError, BedrockError) as exc:
            print(f"[error] {exc}\n")
            continue
        print(f"     {result.outcome.explain()}")
        print(f"bot> {result.reply.strip()}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisis-triage",
        description="Crisis triage agent: Decider 2B for tone, Amazon Bedrock for safety and replies.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--health", action="store_true", help="report Decider and Bedrock status")
    mode.add_argument("--test", action="store_true", help="run the fixture set")
    mode.add_argument("--demo", action="store_true", help="run the scripted demo conversation")
    mode.add_argument("--interactive", action="store_true", help="talk to the agent")

    parser.add_argument(
        "--low",
        type=float,
        default=None,
        help="P below this routes to calm (default: $CRISIS_LOW; no built-in default)",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help="P at or above this routes to crisis (default: $CRISIS_HIGH; no built-in default)",
    )
    parser.add_argument("--esc", type=float, default=None, help="escalation threshold (default 0.70)")
    parser.add_argument("--max-turns", type=int, default=6, help="user turns kept in Decider's state")
    parser.add_argument(
        "--no-safety-check",
        action="store_true",
        help="disable the Bedrock safety check (NOT recommended; Decider alone misses emergencies)",
    )
    parser.add_argument("--out", type=str, default=None, help="write results to this JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not any([args.health, args.test, args.demo, args.interactive]):
        parser.print_help()
        return 0

    try:
        if args.health:
            return cmd_health(args)
        if args.test:
            return cmd_test(args)
        if args.demo:
            return cmd_demo(args)
        return cmd_interactive(args)
    except BandPolicyError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())