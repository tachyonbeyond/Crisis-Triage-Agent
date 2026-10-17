"""Sweep the fixture set through the live Decider server and measure separation.

Purpose: the documented bands (HIGH=0.75 / LOW=0.25) were guesses. The model's real
output scale turned out to be compressed, so the thresholds must come from measured
separability on our own fixtures. This script produces that measurement.

Run against a live Decider server (terminal A):

    .\\.venv\\Scripts\\python.exe scripts/sweep_fixtures.py

Writes results/fixtures_decider.json. Every number quoted in docs/README must come
from that file -- never from memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "test_cases.json"
QUESTIONS = REPO / "scripts" / "fixtures" / "systemone_smoke.json"
OUT = REPO / "results" / "fixtures_decider.json"

DEFAULT_URL = "http://127.0.0.1:8000"


def build_state(turns: list[str], max_turns: int = 6) -> str:
    """The state we send to Decider: the last N *user* turns, oldest first.

    Labelled so the `escalating` question has something to compare against, and
    capped so we stay well inside the model's 4096-token window.
    """
    recent = turns[-max_turns:]
    if len(recent) == 1:
        return recent[0]
    return "\n".join(f"USER: {t}" for t in recent)


def post(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/systemone", data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, (time.perf_counter() - started) * 1000.0


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]

    # Health first, so a down server fails loudly instead of per-case.
    health = None
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as r:
            health = json.load(r)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"Decider server not reachable at {args.url}: {exc!r}")
        return 2

    print(
        f"server: {health['model']}  device={health['device']}  "
        f"num_slots={health['num_slots']}  max_length={health['max_length']}"
    )
    print(f"sweeping {len(cases)} fixtures ...\n")

    # Warm the model so the first case is not a cold-start outlier.
    try:
        post(args.url, {"state": "warmup", "questions": questions}, args.timeout)
    except Exception:  # noqa: BLE001
        pass

    rows: list[dict] = []
    for case in cases:
        state = build_state(case["turns"])
        payload = {"state": state, "questions": questions}
        try:
            out, wall_ms = post(args.url, payload, args.timeout)
            answers = out["answers"]
            row = {
                "id": case["id"],
                "label": case["label"],
                "group": case["group"],
                "turns": len(case["turns"]),
                "state": state,
                "is_emergency": answers["is_emergency"]["noul"],
                "escalating": answers["escalating"]["noul"],
                "crisis_type": answers["crisis_type"]["choice"],
                "crisis_type_probs": answers["crisis_type"]["probabilities"],
                "crisis_type_confidence": answers["crisis_type"]["confidence"],
                "urgency_score": answers["urgency_level"]["score"],
                "urgency_probs": answers["urgency_level"]["probabilities"],
                "urgency_confidence": answers["urgency_level"]["confidence"],
                "input_tokens": out["usage"]["input_tokens"],
                "server_latency_ms": out["latency_ms"],
                "wall_latency_ms": round(wall_ms, 2),
                "ok": True,
            }
        except Exception as exc:  # noqa: BLE001 - a failed case is data
            row = {
                "id": case["id"],
                "label": case["label"],
                "group": case["group"],
                "turns": len(case["turns"]),
                "state": state,
                "ok": False,
                "error": repr(exc),
            }
        rows.append(row)
        if row["ok"]:
            print(
                f"  {row['id']:<28} p={row['is_emergency']:.4f}  esc={row['escalating']:.4f}  "
                f"type={row['crisis_type']:<20} urg={row['urgency_score']:.2f}  "
                f"{row['server_latency_ms']:>7.1f} ms"
            )
        else:
            print(f"  {row['id']:<28} FAILED {row['error']}")

    ok = [r for r in rows if r["ok"]]
    by_label: dict[str, list[float]] = {}
    for r in ok:
        by_label.setdefault(r["label"], []).append(r["is_emergency"])

    print("\n--- is_emergency by intended label ---")
    stats: dict[str, dict] = {}
    for label in sorted(by_label):
        vals = by_label[label]
        s = {
            "n": len(vals),
            "min": round(min(vals), 4),
            "median": round(statistics.median(vals), 4),
            "max": round(max(vals), 4),
        }
        stats[label] = s
        print(
            f"  {label:<9} n={s['n']:<3} min={s['min']:.4f}  median={s['median']:.4f}  max={s['max']:.4f}"
        )

    # Where could a HIGH threshold sit so crisis outranks everything else?
    sep = None
    if by_label.get("crisis") and by_label.get("calm"):
        crisis_vals = sorted(by_label["crisis"])
        calm_vals = sorted(by_label["calm"])
        other = [v for lab, vs in by_label.items() if lab != "crisis" for v in vs]
        sep = {
            "crisis_min": round(crisis_vals[0], 4),
            "non_crisis_max": round(max(other), 4),
            "overlap": max(other) >= crisis_vals[0],
            "calm_max": round(calm_vals[-1], 4),
            "crisis_median": round(statistics.median(crisis_vals), 4),
            "non_crisis_median": round(statistics.median(other), 4),
        }
        print("\n--- separation ---")
        for k, v in sep.items():
            print(f"  {k}: {v}")
        if sep["overlap"]:
            print(
                "  WARNING: crisis and non-crisis distributions overlap -- no single "
                "threshold separates them. Inspect the misordered cases before tuning."
            )

    lat = [r["server_latency_ms"] for r in ok]
    wall = [r["wall_latency_ms"] for r in ok]
    latency = {
        "server_ms": {
            "p50": round(percentile(lat, 0.50), 2),
            "p95": round(percentile(lat, 0.95), 2),
            "min": round(min(lat), 2),
            "max": round(max(lat), 2),
        },
        "wall_ms": {
            "p50": round(percentile(wall, 0.50), 2),
            "p95": round(percentile(wall, 0.95), 2),
        },
        "input_tokens": {
            "min": min(r["input_tokens"] for r in ok),
            "max": max(r["input_tokens"] for r in ok),
        },
    }
    print(
        f"\nserver latency p50={latency['server_ms']['p50']} ms  p95={latency['server_ms']['p95']} ms"
    )

    result = {
        "generated_with": "scripts/sweep_fixtures.py",
        "fixtures_file": str(FIXTURES.relative_to(REPO)).replace("\\", "/"),
        "questions_file": str(QUESTIONS.relative_to(REPO)).replace("\\", "/"),
        "hardware": {
            "note": "Decider served locally on this machine; see docs/GROUND_TRUTH.md",
            "server_health": health,
        },
        "n_cases": len(rows),
        "n_ok": len(ok),
        "by_label": stats,
        "separation": sep,
        "latency": latency,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())