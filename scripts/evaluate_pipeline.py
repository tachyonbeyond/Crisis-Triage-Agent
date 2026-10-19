"""Evaluate the full triage pipeline over a fixtures file and report per-case routes.

Why this exists: the headline claim -- "the Bedrock safety gate catches the emergencies
Decider's band misses" -- was assembled by hand from three separate files, and nothing in
scripts/ could regenerate it. This script runs the *production* path end to end, once per
case: `CrisisTriagePolicy.triage(case["turns"])`, the same call `crisis_triage.main` makes
and the same configuration the Strands handler is built with
(`crisis_triage.main._build_agent`). The OR rule lives in the policy; it is never
re-implemented here. This file only observes and reports what the policy returned.

Run against a live Decider server (terminal A) and real Bedrock credentials:

    .venv/Scripts/python.exe scripts/evaluate_pipeline.py

    # a held-out set, evaluated once, with the guard that implies
    .venv/Scripts/python.exe scripts/evaluate_pipeline.py --fixtures tests/fixtures/heldout_cases.json --confirm-heldout

Writes results/pipeline_test_cases.json. Every full-pipeline number quoted in docs/ or the
README must come from that file -- never from memory. Fixtures are opened read-only; this
script never edits them. It never reads or logs the contents of .env: only whether the
Bedrock key is present, as a boolean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from crisis_triage.bedrock.safety_check import SafetyChecker  # noqa: E402
from crisis_triage.decider_client import DeciderClient  # noqa: E402
from crisis_triage.policies.confidence_band import (  # noqa: E402
    BandPolicyError,
    ConfidenceBandPolicy,
    Route,
)
from crisis_triage.policies.crisis_policy import (  # noqa: E402
    DEFAULT_ESC_THRESHOLD,
    CrisisTriagePolicy,
)
from crisis_triage.policies.questions import (  # noqa: E402
    CRISIS_QUESTIONS,
    DEFAULT_MAX_TURNS,
    build_state,
)

FIXTURES = REPO / "tests" / "fixtures" / "test_cases.json"
RESULTS = REPO / "results"
DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 120.0

CRISIS = Route.CRISIS.value
CHECK_IN = Route.CHECK_IN.value
CALM = Route.CALM.value
LABEL_ORDER = [CRISIS, CHECK_IN, CALM]

# The group the "clear emergency" fixtures carry in tests/fixtures/test_cases.json.
CLEAR_EMERGENCY = "clear_emergency"

DEV_NOTE = (
    "Thresholds and the safety prompt were developed using these cases; "
    "results are optimistic."
)

DEFINITIONS = {
    "crisis_label": "a case whose label field is 'crisis'",
    "non_crisis_label": "a case whose label field is anything else",
    "denominators": (
        "k/n denominators are the TOTAL labelled counts in the fixtures file, not the "
        "number of cases that produced a route. A case that errored without a route "
        "therefore counts as a crisis miss and as incorrect -- the conservative direction."
    ),
    "crisis_recall": (
        "crisis-labelled cases whose FINAL route is crisis. A miss here means an "
        "emergency was routed below crisis: the failure that matters."
    ),
    "emergencies_below_crisis": (
        "crisis-labelled cases whose FINAL route is not crisis (check_in, calm, or a case "
        "that errored with no route)."
    ),
    "false_escalations": (
        "non-crisis-labelled cases whose FINAL route is crisis. Each one sends crisis "
        "resources to someone who was not labelled as needing them."
    ),
    "correct_strict": "cases whose FINAL route string equals the label string.",
    "correct_lenient": (
        "as crisis_triage.main --test counts agreement: strict, plus a check_in label also "
        "accepting a crisis route. Reported as a secondary figure only."
    ),
    "signals": (
        "each row is ONE signal alone, over the whole file, with the same denominators as "
        "the final pipeline row. crisis_caught = crisis-labelled cases the signal fires on "
        "(k) out of all crisis-labelled cases (n). non_crisis_flagged = non-crisis-labelled "
        "cases it fires on (k) out of all non-crisis-labelled cases (n)."
    ),
    "safety_gate_alone": (
        "safety_emergency is True. Includes the fail-closed verdict returned when the "
        "safety check itself errored (safety_failed)."
    ),
    "band_alone": (
        "band_route is crisis -- Decider's high band. Includes the fail-safe fallback that "
        "routes to crisis when Decider is unreachable."
    ),
    "escalating_alone": "p_escalating >= the escalating threshold.",
    "band_or_escalating": "band_alone OR escalating_alone.",
    "final_pipeline": "the final route is crisis -- what the product actually does.",
    "crisis_caught_only_by_gate": (
        "crisis-labelled cases with FINAL route crisis AND band_route not crisis AND "
        "escalating below threshold AND safety_emergency True. These are the cases only the "
        "Bedrock gate catches; without it they would have been routed below crisis."
    ),
    "non_crisis_raised_only_by_gate": "the mirror: non-crisis cases whose only trigger was the gate.",
    "n_errored": (
        "cases where an unexpected exception escaped the production triage path. route is "
        "null for those and they still count in every denominator above."
    ),
    "n_fail_safe": (
        "cases where the PRODUCTION fail-safe fired: Decider unreachable (fallback_reason "
        "set) or the safety check failed (fail-closed). Routing to crisis there is by "
        "design, not an error, so it is counted separately from n_errored."
    ),
    "n_state_mismatches": (
        "cases where the string sent to Decider differs from the string sent to the safety "
        "check. Production builds one string for both; anything but 0 means that guarantee "
        "broke. A case with no safety checker records null and is not counted."
    ),
    "n_build_state_mismatches": (
        "cases where the recorded state differs from "
        "crisis_triage.policies.questions.build_state(turns, max_turns)."
    ),
    "safety_failed_flag": (
        "safety_failed is True when the safety check could not reach Bedrock and returned "
        "its fail-closed verdict instead of a real classification."
    ),
    "latency_ms": "per-leg wall latency as reported by the server/client, warm-up excluded.",
}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def build_policy(
    decider,
    safety,
    *,
    low: float | None = None,
    high: float | None = None,
    esc_threshold: float | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> CrisisTriagePolicy:
    """The handler's configuration, byte for byte -- see crisis_triage.main._build_agent.

    low=None / high=None make ConfidenceBandPolicy read CRISIS_LOW/CRISIS_HIGH from the
    environment (it has no built-in default, on purpose), and esc_threshold=None makes
    CrisisTriagePolicy read CRISIS_ESC (default 0.70). Nothing here invents a threshold.
    """
    band = ConfidenceBandPolicy(decider, low=low, high=high, max_turns=max_turns)
    return CrisisTriagePolicy(band, safety, esc_threshold=esc_threshold)


def make_decider(url: str, timeout_s: float) -> DeciderClient:
    """Factory, so tests can inject a double instead of a live server."""
    return DeciderClient(base_url=url, timeout_s=timeout_s)


def make_safety() -> SafetyChecker:
    """Factory, so tests can inject a double instead of a real Bedrock client."""
    return SafetyChecker()


def _env_var(name: str) -> str | None:
    raw = os.environ.get(name)
    return f"environment:{name}" if raw is not None and raw.strip() else None


def threshold_report(policy: CrisisTriagePolicy) -> dict:
    """The thresholds actually in force, and where each one came from."""
    band = policy.band_policy
    return {
        "low": band.low,
        "low_source": _env_var("CRISIS_LOW") or "constructor argument",
        "high": band.high,
        "high_source": _env_var("CRISIS_HIGH") or "constructor argument",
        "escalating": policy.esc_threshold,
        "escalating_source": _env_var("CRISIS_ESC") or f"built-in default {DEFAULT_ESC_THRESHOLD}",
        "max_turns": band.max_turns,
        "note": (
            "These control tone and the escalation signal only. Safety routing is an OR "
            "and never depends on a threshold."
        ),
    }


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def is_heldout(path: Path) -> bool:
    """A held-out fixtures file is identified by its path, by choice."""
    return "heldout" in str(path).replace("\\", "/").lower()


def out_guard(
    out_path: Path,
    *,
    force: bool,
    heldout: bool,
    confirm_heldout: bool,
) -> str | None:
    """Return a refusal reason, or None when it is safe to write out_path.

    Three rules, in order:
      1. a held-out set is evaluated once, so it needs --confirm-heldout and its output
         must not already exist -- --force is deliberately not enough to clobber it;
      2. any other existing output needs --force;
      3. otherwise fine.
    """
    if heldout:
        if not confirm_heldout:
            return (
                "held-out fixtures: pass --confirm-heldout to run them. They are evaluated "
                "once and never tuned on, and the output must not already exist."
            )
        if out_path.exists():
            return (
                f"{out_path} already exists -- the held-out set may be evaluated only once. "
                "Move or delete the old file deliberately; --force will not overwrite it."
            )
        return None
    if out_path.exists() and not force:
        return f"{out_path} already exists; pass --force to overwrite"
    return None


# ---------------------------------------------------------------------------
# reading and running
# ---------------------------------------------------------------------------


def load_fixtures(path: Path) -> tuple[dict, str]:
    """Read the fixtures file read-only. Returns (document, sha256 of the exact bytes)."""
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _finite(value):
    """JSON-safe number: NaN, inf and None all become None so the output is valid JSON."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(REPO), capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict:
    """The commit the run belongs to, and whether the tree was dirty at run time."""
    porcelain = _git(["status", "--porcelain"]) or ""
    dirty_files = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "git_dirty_files": dirty_files,
    }


def evaluate_case(policy: CrisisTriagePolicy, case: dict, max_turns: int) -> dict:
    """Run one case through the production triage path and record the audit trail.

    No routing logic lives here. policy.triage(turns) builds the canonical state and
    applies the OR rule; this function only reads the returned TriageOutcome.
    """
    turns = list(case.get("turns") or [])
    row: dict = {
        "id": case.get("id"),
        "label": case.get("label"),
        "group": case.get("group"),
        "n_turns": len(turns),
        "turns": turns,
        "state": None,
        "state_matches_build_state": None,
        "state_same_for_both": None,
        "p_emergency": None,
        "p_escalating": None,
        "band_route": None,
        "safety_emergency": None,
        "safety_reason": None,
        "safety_failed": None,
        "route": None,
        "triggered_by": [],
        "raised_by": "",
        "fallback_reason": None,
        "decider_latency_ms": None,
        "safety_latency_ms": None,
        "decider_input_tokens": None,
        "decider_output_tokens": None,
        "safety_input_tokens": None,
        "safety_output_tokens": None,
        "error": None,
    }

    try:
        expected_state = build_state(turns, max_turns)
    except ValueError as exc:
        # Production fails here too: its triage() calls build_state first.
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        outcome = policy.triage(turns)
    except Exception as exc:  # noqa: BLE001 - a broken case is data, not a crash
        # Production's fail-safe lives inside the policy: decide_or_fallback handles a
        # dead Decider and SafetyChecker.check handles a dead model, both without raising.
        # Reaching here means something genuinely unexpected happened, so the case is
        # recorded with an error and NO route rather than dropped or silently routed.
        row["state"] = expected_state
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    band = outcome.band
    safety = outcome.safety
    state = band.state if band.state is not None else expected_state

    row.update(
        {
            "state": state,
            "state_matches_build_state": state == expected_state,
            "state_same_for_both": None if safety is None else state == outcome.safety_state,
            "p_emergency": _finite(band.probability),
            "p_escalating": _finite(band.escalating_probability),
            "band_route": band.route.value,
            "safety_emergency": None if safety is None else bool(safety.emergency),
            "safety_reason": None if safety is None else safety.reason,
            "safety_failed": None if safety is None else bool(safety.failed),
            "route": outcome.route.value,
            "triggered_by": list(outcome.triggered_by),
            "raised_by": "+".join(outcome.triggered_by),
            "fallback_reason": band.fallback_reason,
            "decider_latency_ms": _finite(band.decider_latency_ms),
            "safety_latency_ms": None if safety is None else _finite(safety.latency_ms),
            "decider_input_tokens": band.input_tokens,
            "decider_output_tokens": band.output_tokens,
            "safety_input_tokens": 0 if safety is None else safety.input_tokens,
            "safety_output_tokens": 0 if safety is None else safety.output_tokens,
        }
    )
    return row


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


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


def _counts(k: int, n: int, ids: list) -> dict:
    return {"k": k, "n": n, "text": f"{k}/{n}", "ids": list(ids)}


def _signal_row(rows: list[dict], predicate) -> dict:
    """One signal alone: how many crisis cases it catches, how many others it flags."""
    crisis = [r for r in rows if r.get("label") == CRISIS]
    non_crisis = [r for r in rows if r.get("label") != CRISIS]
    caught = [r for r in crisis if predicate(r)]
    flagged = [r for r in non_crisis if predicate(r)]
    return {
        "crisis_caught": f"{len(caught)}/{len(crisis)}",
        "crisis_caught_k": len(caught),
        "crisis_caught_n": len(crisis),
        "crisis_missed_ids": [r["id"] for r in crisis if not predicate(r)],
        "non_crisis_flagged": f"{len(flagged)}/{len(non_crisis)}",
        "non_crisis_flagged_k": len(flagged),
        "non_crisis_flagged_n": len(non_crisis),
        "false_escalation_ids": [r["id"] for r in flagged],
    }


def _lenient_agrees(row: dict) -> bool:
    """How crisis_triage.main --test counts agreement: check_in also accepts crisis."""
    label = row.get("label")
    route = row.get("route")
    if label is None or route is None:
        return False
    return route == label or (label == CHECK_IN and route in {CHECK_IN, CRISIS})


def _confusion(rows: list[dict]) -> dict:
    present = {r.get("label") for r in rows}
    labels = [lab for lab in LABEL_ORDER if lab in present]
    labels += sorted(lab for lab in present if lab not in LABEL_ORDER and lab is not None)
    routes = [CRISIS, CHECK_IN, CALM, "error"]
    counts = {lab: {route: 0 for route in routes} for lab in labels}
    for row in rows:
        label = row.get("label")
        route = row.get("route")
        if label not in counts:
            continue
        counts[label][route if route in (CRISIS, CHECK_IN, CALM) else "error"] += 1
    return {
        "labels": labels,
        "routes": routes,
        "counts": counts,
        "note": (
            "rows are the intended label, columns are the FINAL route. The 'error' column "
            "holds cases whose route is null because the triage path raised."
        ),
    }


def summarize(rows: list[dict], *, esc_threshold: float) -> dict:
    """Every headline metric, computed from the recorded rows. Pure arithmetic."""
    crisis_rows = [r for r in rows if r.get("label") == CRISIS]
    non_crisis_rows = [r for r in rows if r.get("label") != CRISIS]

    def gate(r: dict) -> bool:
        return r.get("safety_emergency") is True

    def band(r: dict) -> bool:
        return r.get("band_route") == CRISIS

    def escalating(r: dict) -> bool:
        value = r.get("p_escalating")
        return value is not None and value >= esc_threshold

    def band_or_esc(r: dict) -> bool:
        return band(r) or escalating(r)

    def final(r: dict) -> bool:
        return r.get("route") == CRISIS

    caught = [r for r in crisis_rows if final(r)]
    missed = [r for r in crisis_rows if not final(r)]
    false_esc = [r for r in non_crisis_rows if final(r)]
    strict_ok = [r for r in rows if r.get("route") == r.get("label")]
    lenient_ok = [r for r in rows if _lenient_agrees(r)]

    def only_gate(r: dict) -> bool:
        return final(r) and not band(r) and not escalating(r) and gate(r)

    clear = [r for r in crisis_rows if r.get("group") == CLEAR_EMERGENCY]
    gate_only_crisis = [r for r in crisis_rows if only_gate(r)]
    gate_only_clear = [r for r in clear if only_gate(r)]
    gate_only_non_crisis = [r for r in non_crisis_rows if only_gate(r)]

    errored = [r for r in rows if r.get("error")]
    fail_safe = [r for r in rows if r.get("fallback_reason") or r.get("safety_failed")]
    mismatched = [r for r in rows if r.get("state_same_for_both") is False]
    build_mismatched = [r for r in rows if r.get("state_matches_build_state") is False]

    dec_lat = [r["decider_latency_ms"] for r in rows if r.get("decider_latency_ms") is not None]
    saf_lat = [r["safety_latency_ms"] for r in rows if r.get("safety_latency_ms") is not None]

    return {
        "definitions": DEFINITIONS,
        "n_cases": len(rows),
        "n_crisis_labelled": len(crisis_rows),
        "n_non_crisis_labelled": len(non_crisis_rows),
        "confusion_label_x_route": _confusion(rows),
        "crisis_recall": _counts(len(caught), len(crisis_rows), [r["id"] for r in caught]),
        "emergencies_below_crisis": _counts(
            len(missed), len(crisis_rows), [r["id"] for r in missed]
        ),
        "false_escalations": _counts(
            len(false_esc), len(non_crisis_rows), [r["id"] for r in false_esc]
        ),
        "correct_strict": _counts(len(strict_ok), len(rows), [r["id"] for r in strict_ok]),
        "correct_lenient": _counts(len(lenient_ok), len(rows), [r["id"] for r in lenient_ok]),
        "signals": {
            "safety_gate_alone": _signal_row(rows, gate),
            "band_alone": _signal_row(rows, band),
            "escalating_alone": _signal_row(rows, escalating),
            "band_or_escalating": _signal_row(rows, band_or_esc),
            "final_pipeline": _signal_row(rows, final),
        },
        "crisis_caught_only_by_gate": {
            "all_crisis": _counts(
                len(gate_only_crisis), len(crisis_rows), [r["id"] for r in gate_only_crisis]
            ),
            "clear_emergencies": _counts(
                len(gate_only_clear), len(clear), [r["id"] for r in gate_only_clear]
            ),
            "note": (
                "clear_emergencies selects the crisis cases whose group is "
                f"{CLEAR_EMERGENCY!r}; the other crisis cases are multi-turn ones where the "
                "escalation signal or the band can also fire."
            ),
        },
        "non_crisis_raised_only_by_gate": _counts(
            len(gate_only_non_crisis),
            len(non_crisis_rows),
            [r["id"] for r in gate_only_non_crisis],
        ),
        "n_errored": len(errored),
        "errored_ids": [r["id"] for r in errored],
        "n_fail_safe": len(fail_safe),
        "fail_safe_ids": [r["id"] for r in fail_safe],
        "n_state_mismatches": len(mismatched),
        "state_mismatch_ids": [r["id"] for r in mismatched],
        "n_build_state_mismatches": len(build_mismatched),
        "build_state_mismatch_ids": [r["id"] for r in build_mismatched],
        "latency_ms": {
            "decider": {
                "n": len(dec_lat),
                "p50": _finite(round(percentile(dec_lat, 0.50), 2)),
                "p95": _finite(round(percentile(dec_lat, 0.95), 2)),
            },
            "safety": {
                "n": len(saf_lat),
                "p50": _finite(round(percentile(saf_lat, 0.50), 2)),
                "p95": _finite(round(percentile(saf_lat, 0.95), 2)),
            },
        },
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _cell(value, spec: str = ".4f") -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _safety_cell(row: dict) -> str:
    if row.get("safety_emergency") is None:
        return "-"
    base = "yes" if row["safety_emergency"] else "no"
    return base + "!" if row.get("safety_failed") else base


def _print_report(report: dict) -> None:
    header = report["header"]
    summary = report["summary"]

    print(f"fixtures : {header['fixtures_path']}")
    print(f"sha256   : {header['fixtures_sha256']}")
    print(f"cases    : {header['n_cases']}   errored: {header['n_errored']}")
    print(f"git      : {header['git_commit'] or '-'}  dirty={header['git_dirty']}")
    thresholds = header["thresholds"]
    print(
        "thresholds: "
        f"low={thresholds['low']} ({thresholds['low_source']})  "
        f"high={thresholds['high']} ({thresholds['high_source']})  "
        f"escalating={thresholds['escalating']} ({thresholds['escalating_source']})  "
        f"max_turns={thresholds['max_turns']}"
    )
    bedrock = header["bedrock"]
    print(
        f"bedrock  : model={bedrock['model_id']}  region={bedrock['region']}  "
        f"key_present={bedrock['api_key_present']}"
    )
    health = header["decider_health"]
    if health.get("ok"):
        print(f"decider  : {health['body']}")
    else:
        print(
            f"decider  : UNREACHABLE ({health.get('error')}) "
            "-- every case fails safe to CRISIS"
        )
    if header["dev_set_note"]:
        print(f"NOTE     : {header['dev_set_note']}")

    print(
        f"\n{'id':<26} {'label':<9} {'band':<9} {'P(em)':>7} {'escal':>7} {'safety':>7} "
        f"{'final':<9} {'raised_by':<24} {'dec_ms':>8} {'saf_ms':>8}"
    )
    for row in report["cases"]:
        print(
            f"{str(row['id']):<26} {str(row['label']):<9} {str(row['band_route']):<9} "
            f"{_cell(row['p_emergency']):>7} {_cell(row['p_escalating']):>7} "
            f"{_safety_cell(row):>7} {str(row['route']):<9} "
            f"{(row['raised_by'] or '-'):<24} "
            f"{_cell(row['decider_latency_ms'], '.0f'):>8} "
            f"{_cell(row['safety_latency_ms'], '.0f'):>8}"
            + (f"   ERROR {row['error']}" if row["error"] else "")
        )
    print("  safety 'yes!' = the safety check failed and returned its fail-closed verdict")

    signals = summary["signals"]
    print("\n--- summary ---")
    print(
        f"  crisis recall           : {summary['crisis_recall']['text']}"
        "   (crisis-labelled cases whose FINAL route is crisis)"
    )
    print(
        f"  emergencies below crisis: {summary['emergencies_below_crisis']['text']}"
        "   (crisis-labelled cases NOT routed to crisis; each one is the dangerous failure)"
    )
    print(
        f"  false escalations       : {summary['false_escalations']['text']}"
        "   (non-crisis-labelled cases routed to crisis)"
    )
    print(
        f"  correct (strict)        : {summary['correct_strict']['text']}"
        "   (FINAL route string equals the label string)"
    )
    print(
        f"  correct (lenient)       : {summary['correct_lenient']['text']}"
        "   (as crisis_triage.main --test counts it; check_in also accepts crisis)"
    )
    print("\n  signal alone            crisis caught   non-crisis flagged")
    for name in (
        "safety_gate_alone",
        "band_alone",
        "escalating_alone",
        "band_or_escalating",
        "final_pipeline",
    ):
        row = signals[name]
        print(
            f"  {name:<23} {row['crisis_caught']:>13} {row['non_crisis_flagged']:>19}"
        )
    print("  (denominators are all crisis-labelled and all non-crisis-labelled cases)")

    gate_only = summary["crisis_caught_only_by_gate"]
    print(
        f"\n  caught only by the gate : {gate_only['all_crisis']['text']} of all crisis cases"
        f"   ({gate_only['clear_emergencies']['text']} of the clear-emergency group)"
    )
    if gate_only["all_crisis"]["ids"]:
        print(f"    ids: {', '.join(gate_only['all_crisis']['ids'])}")
    mirror = summary["non_crisis_raised_only_by_gate"]
    print(f"  raised only by the gate (non-crisis): {mirror['text']}")
    if mirror["ids"]:
        print(f"    ids: {', '.join(mirror['ids'])}")

    print(
        f"\n  errored (no route)      : {summary['n_errored']}"
        f"   fail-safe fired: {summary['n_fail_safe']}"
        f"   state mismatches: {summary['n_state_mismatches']}"
    )
    lat = summary["latency_ms"]
    print(
        f"  latency ms              : decider p50={_cell(lat['decider']['p50'], '.0f')} "
        f"p95={_cell(lat['decider']['p95'], '.0f')} (n={lat['decider']['n']})   "
        f"safety p50={_cell(lat['safety']['p50'], '.0f')} "
        f"p95={_cell(lat['safety']['p95'], '.0f')} (n={lat['safety']['n']})"
    )
    print("  every metric is defined in the 'definitions' block of the written JSON")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run(
    fixtures_path: Path,
    out_path: Path,
    *,
    decider,
    safety,
    low: float | None = None,
    high: float | None = None,
    esc_threshold: float | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    warmup: bool = True,
    write: bool = True,
) -> dict:
    """Evaluate every case through the production path and return the report."""
    fixtures_path = Path(fixtures_path)
    out_path = Path(out_path)

    document, fixtures_sha = load_fixtures(fixtures_path)
    cases = list(document.get("cases") or [])

    policy = build_policy(
        decider, safety, low=low, high=high, esc_threshold=esc_threshold, max_turns=max_turns
    )
    band = policy.band_policy

    if warmup:
        # One throwaway Decider call so the first case is not a cold-start outlier in the
        # latency column. Excluded from the results below.
        try:
            decider.evaluate("warmup", CRISIS_QUESTIONS)
        except Exception:  # noqa: BLE001 - a warm-up must never fail the run
            pass

    health = _decider_health(decider)

    rows = [evaluate_case(policy, case, band.max_turns) for case in cases]
    summary = summarize(rows, esc_threshold=policy.esc_threshold)

    questions = band.questions
    header = {
        "generated_with": "scripts/evaluate_pipeline.py",
        "how_to_rerun": "python scripts/evaluate_pipeline.py [--fixtures PATH] [--out PATH]",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixtures_path": str(fixtures_path),
        "fixtures_sha256": fixtures_sha,
        "n_cases": len(rows),
        "n_errored": summary["n_errored"],
        "thresholds": threshold_report(policy),
        "bedrock": _bedrock_info(safety),
        "decider_health": health,
        "warmup": warmup,
        "questions": {
            "names": list(questions),
            "questions": questions,
            "sha256": hashlib.sha256(
                json.dumps(questions, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "dev_set_note": DEV_NOTE if fixtures_path.name == "test_cases.json" else None,
        **git_state(),
    }

    report = {
        "generated_with": "scripts/evaluate_pipeline.py",
        "header": header,
        "summary": summary,
        "cases": rows,
    }

    _print_report(report)

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")
    return report


def _decider_health(decider) -> dict:
    """The Decider /health body, or why we could not get it. Never raises."""
    try:
        body = decider.health(timeout_s=10.0)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must report, never throw
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "body": body}


def _bedrock_info(safety) -> dict:
    """Model and region in use. The key is reported as a boolean and never read out."""
    client = getattr(safety, "client", None)
    return {
        "model_id": getattr(client, "model_id", None),
        "region": getattr(client, "region", None),
        "api_key_present": os.environ.get("AWS_BEARER_TOKEN_BEDROCK") is not None,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=FIXTURES,
        help="fixtures JSON file (default: tests/fixtures/test_cases.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default: results/pipeline_<fixtures stem>.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing --out (never allowed for a held-out fixtures file)",
    )
    parser.add_argument(
        "--confirm-heldout",
        action="store_true",
        help="required when the fixtures path contains 'heldout'",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Decider base URL (default: $DECIDER_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Decider request timeout in seconds (default: 120)",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    fixtures_path = Path(args.fixtures)
    if not fixtures_path.exists():
        print(f"fixtures not found: {fixtures_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else RESULTS / f"pipeline_{fixtures_path.stem}.json"
    reason = out_guard(
        out_path,
        force=args.force,
        heldout=is_heldout(fixtures_path),
        confirm_heldout=args.confirm_heldout,
    )
    if reason:
        print(f"refusing: {reason}", file=sys.stderr)
        return 2

    url = args.url or os.environ.get("DECIDER_URL") or DEFAULT_URL
    decider = make_decider(url, args.timeout)
    safety = make_safety()

    try:
        run(fixtures_path, out_path, decider=decider, safety=safety, warmup=True, write=True)
    except BandPolicyError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
