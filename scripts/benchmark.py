"""Benchmark: Decider 2B vs a Bedrock LLM doing the same classification.

Answers the question a judge will ask -- "why a 2B decision model here?" -- with
measured numbers instead of assertion. Each fixture is classified by:

  a) Decider 2B, local, on the GPU (the `is_emergency` + `escalating` questions)
  b) Bedrock, one prompt per case, JSON answer

and we report latency (p50/p95), token usage, a cost estimate from Bedrock's own
reported token counts, and agreement between the two. Decider's cost is $0 in API
terms (it runs locally); we report GPU time instead of pretending it is free.

    .\\.venv\\Scripts\\python.exe scripts/benchmark.py

Writes results/benchmark.json. Only numbers from that file may be quoted.

Pricing is NOT hardcoded from memory -- pass it in, or read it from the environment.
The script will report tokens and compute cost only when a price is supplied
(docs/DECIDER.md section 5).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "test_cases.json"
QUESTIONS = REPO / "scripts" / "fixtures" / "systemone_smoke.json"
OUT = REPO / "results" / "benchmark.json"

CLASSIFIER_SYSTEM = (
    "You are a crisis-triage classifier. Given the user's message, answer with ONLY a "
    'JSON object: {"emergency": true|false}. emergency=true means imminent danger to self '
    "or others, an active suicide or violence plan, means or a timeline present, or a "
    "medical emergency needing 911. Otherwise false. No prose."
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def build_state(turns: list[str], max_turns: int = 6) -> str:
    recent = turns[-max_turns:]
    return recent[0] if len(recent) == 1 else "\n".join(f"USER: {t}" for t in recent)


def run_decider(url: str, cases: list[dict], questions: dict, timeout: float) -> list[dict]:
    rows = []
    for case in cases:
        body = json.dumps({"state": build_state(case["turns"]), "questions": questions}).encode()
        request = urllib.request.Request(
            f"{url}/v1/systemone", data=body, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        wall_ms = (time.perf_counter() - started) * 1000.0
        answers = payload["answers"]
        rows.append(
            {
                "id": case["id"],
                "expected": case["label"],
                "p_emergency": answers["is_emergency"]["noul"],
                "p_escalating": answers["escalating"]["noul"],
                # Decider's own verdict, not ours: high band OR escalation.
                "decider_emergency": bool(
                    answers["is_emergency"]["noul"] >= 0.30 or answers["escalating"]["noul"] >= 0.70
                ),
                "latency_ms": payload["latency_ms"],
                "wall_ms": round(wall_ms, 2),
                "input_tokens": payload["usage"]["input_tokens"],
                "output_tokens": payload["usage"]["output_tokens"],
            }
        )
    return rows


def run_bedrock(cases: list[dict]) -> list[dict]:
    from crisis_triage.bedrock.client import BedrockError, BedrockTextClient

    client = BedrockTextClient()
    rows = []
    for case in cases:
        state = build_state(case["turns"])
        started = time.perf_counter()
        try:
            parsed, completion = client.complete_json(
                state, system=CLASSIFIER_SYSTEM, max_tokens=60
            )
            verdict = parsed.get("emergency")
            verdict = verdict if isinstance(verdict, bool) else None
            error = None if verdict is not None else "unparseable verdict"
            latency = completion.latency_ms
            in_tokens, out_tokens = completion.input_tokens, completion.output_tokens
        except BedrockError as exc:
            verdict, error = None, f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - started) * 1000.0
            in_tokens = out_tokens = 0
        rows.append(
            {
                "id": case["id"],
                "expected": case["label"],
                "llm_emergency": verdict,
                "latency_ms": round(latency, 2),
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "error": error,
            }
        )
    return rows


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("DECIDER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--input-price",
        type=float,
        default=float(os.environ.get("BEDROCK_INPUT_USD_PER_MTOK", 0) or 0),
        help="USD per million input tokens. Supply from the live Bedrock pricing page.",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=float(os.environ.get("BEDROCK_OUTPUT_USD_PER_MTOK", 0) or 0),
        help="USD per million output tokens.",
    )
    args = parser.parse_args()

    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]

    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as r:
            health = json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"Decider not reachable at {args.url}: {exc!r}")
        return 2

    print(f"Decider: {health['model']} on {health['device']} ({len(cases)} cases)")
    print("running Bedrock classifier ...")
    bedrock_rows = run_bedrock(cases)
    print("running Decider ...")
    decider_rows = run_decider(args.url, cases, questions, args.timeout)

    by_id = {row["id"]: row for row in bedrock_rows}
    merged = []
    for row in decider_rows:
        other = by_id[row["id"]]
        expected_crisis = row["expected"] == "crisis"
        merged.append(
            {
                **row,
                "llm_emergency": other["llm_emergency"],
                "llm_latency_ms": other["latency_ms"],
                "llm_input_tokens": other["input_tokens"],
                "llm_output_tokens": other["output_tokens"],
                "llm_error": other["error"],
                "decider_agrees_with_llm": row["decider_emergency"] == other["llm_emergency"],
                "decider_correct": row["decider_emergency"] == expected_crisis,
                "llm_correct": other["llm_emergency"] == expected_crisis,
            }
        )

    d_lat = [r["latency_ms"] for r in decider_rows]
    l_lat = [r["latency_ms"] for r in bedrock_rows if r["error"] is None]
    in_tokens = sum(r["input_tokens"] for r in bedrock_rows)
    out_tokens = sum(r["output_tokens"] for r in bedrock_rows)
    n = len(merged)

    agreement = sum(1 for r in merged if r["decider_agrees_with_llm"])
    decider_correct = sum(1 for r in merged if r["decider_correct"])
    llm_correct = sum(1 for r in merged if r["llm_correct"])
    crisis = [r for r in merged if r["expected"] == "crisis"]
    decider_recall = sum(1 for r in crisis if r["decider_emergency"])
    llm_recall = sum(1 for r in crisis if r["llm_emergency"])

    cost = None
    if args.input_price or args.output_price:
        cost = {
            "input_usd": round(in_tokens / 1_000_000 * args.input_price, 6),
            "output_usd": round(out_tokens / 1_000_000 * args.output_price, 6),
            "usd_per_1000_turns": round(
                (in_tokens + out_tokens) / n / 1000 * 0, 6
            ),  # replaced below
            "prices_used": {
                "input_usd_per_mtok": args.input_price,
                "output_usd_per_mtok": args.output_price,
            },
        }
        per_turn = in_tokens / n / 1_000_000 * args.input_price + out_tokens / n / 1_000_000 * args.output_price
        cost["usd_per_1000_turns"] = round(per_turn * 1000, 6)

    summary = {
        "n_cases": n,
        "decider": {
            "latency_p50_ms": round(percentile(d_lat, 0.50), 2),
            "latency_p95_ms": round(percentile(d_lat, 0.95), 2),
            "median_input_tokens": statistics.median(
                [r["input_tokens"] for r in decider_rows]
            ),
            "correct": f"{decider_correct}/{n}",
            "crisis_recall": f"{decider_recall}/{len(crisis)}",
            "api_cost_usd": 0.0,
            "note": "runs locally on the GPU; no API cost, but the GPU is busy ~p50 ms per call",
        },
        "bedrock_llm": {
            "latency_p50_ms": round(percentile(l_lat, 0.50), 2),
            "latency_p95_ms": round(percentile(l_lat, 0.95), 2),
            "total_input_tokens": in_tokens,
            "total_output_tokens": out_tokens,
            "correct": f"{llm_correct}/{n}",
            "crisis_recall": f"{llm_recall}/{len(crisis)}",
            "cost": cost,
            "note": "cost omitted unless --input-price/--output-price are supplied from the pricing page",
        },
        "agreement": {
            "n": agreement,
            "of": n,
            "rate": round(agreement / n, 4),
        },
    }

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))

    out = {
        "generated_with": "scripts/benchmark.py",
        "hardware": {
            "decider_device": health.get("device"),
            "decider_model": health.get("model"),
            "note": "see docs/GROUND_TRUTH.md for the machine",
        },
        "bedrock_model_id": os.environ.get("BEDROCK_MODEL_ID"),
        "bedrock_region": os.environ.get("AWS_DEFAULT_REGION"),
        "summary": summary,
        "cases": merged,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())