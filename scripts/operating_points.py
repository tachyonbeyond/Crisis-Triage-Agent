"""Operating points and AUC for Decider's scores on the 22 labelled fixtures.

Reads ONLY `results/fixtures_decider.json` (the saved sweep), so the table is
reproducible offline and cannot drift from the committed results. Standard library
only; no server, no credentials.

    .\\venv\\Scripts\\python.exe scripts/operating_points.py

Definitions (also written into the JSON):

- positives = label "crisis" (n=8); negatives = the other 14 (calm 9 + check_in 5).
  This is the binary crisis-vs-non-crisis split.
- flag = score >= t. recall = TP/8. false escalations = FP/14. precision = TP/(TP+FP).
- AUC = probability that a random positive outranks a random negative, computed by
  average ranks with ties shared (Mann-Whitney U), and independently cross-checked
  by brute-force pair counting.

Writes results/operating_points.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "results" / "fixtures_decider.json"
DEFAULT_OUT = REPO / "results" / "operating_points.json"

# The values that matter for the story, always shown even if a sweep row collapses.
CURRENT_HIGH = 0.30
CURRENT_LOW = 0.15
DOCUMENTED_HIGH = 0.75
ESC_THRESHOLD = 0.70

# The 6 clear emergencies.
CLEAR_EMERGENCIES = {
    "em_plan_means",
    "em_overdose_now",
    "em_weapon_partner",
    "em_bridge_now",
    "em_third_party_note",
    "em_chest_pain",
}
# The 6 unambiguous calm controls, and a harder variant that adds the passive-ideation
# edge case (a check_in, i.e. genuinely ambiguous). Reported separately so the doc's
# separation claim names exactly which controls it rests on.
CLEAR_CONTROLS = {
    "calm_billing",
    "calm_exam_stress",
    "calm_breakup",
    "calm_grief",
    "calm_panic_safe",
    "calm_job_search",
}
HARD_CONTROLS = {"edge_passive_ideation"}

# Which shipped threshold belongs to which question, so the table is not mislabelled.
FIELD_REFERENCES = {"is_emergency": {DOCUMENTED_HIGH, CURRENT_HIGH, CURRENT_LOW}, "escalating": {ESC_THRESHOLD}}

FIELD_LABELS = {
    ("is_emergency", DOCUMENTED_HIGH): "superseded documented bar",
    ("is_emergency", CURRENT_HIGH): "shipped CRISIS_HIGH (tone)",
    ("is_emergency", CURRENT_LOW): "shipped CRISIS_LOW (tone)",
    ("escalating", ESC_THRESHOLD): "shipped CRISIS_ESC",
}


def auc_rank(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC via average ranks (ties share the mean rank)."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [ranks[i] for i in range(n) if labels[i] == 1]
    n1 = len(pos)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs both classes")
    return (sum(pos) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_bruteforce(scores: list[float], labels: list[int]) -> float:
    """Independent check: fraction of positive-negative pairs correctly ordered."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def sweep(cases: list[dict], field: str) -> list[dict]:
    """One row per distinct score value, plus reference rows, runs collapsed."""
    ordered = sorted(cases, key=lambda c: -c[field])
    n_pos = sum(1 for c in cases if c["label"] == "crisis")
    n_neg = len(cases) - n_pos

    thresholds = sorted({c[field] for c in cases})
    references = set(FIELD_REFERENCES[field])
    # Add the score of the lowest crisis case: the cheapest threshold that catches all.
    crisis_scores = [c[field] for c in cases if c["label"] == "crisis"]
    if crisis_scores:
        references.add(min(crisis_scores))
    for r in references:
        thresholds.append(round(r, 4))
    thresholds = sorted(set(thresholds))

    rows: list[dict] = []
    seen: dict[tuple, int] = {}
    for t in thresholds:
        flagged = [c for c in cases if c[field] >= t]
        tp = sum(1 for c in flagged if c["label"] == "crisis")
        fp = len(flagged) - tp
        fn = n_pos - tp
        row = {
            "threshold": round(t, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": f"{tp}/{n_pos}",
            "false_escalations": f"{fp}/{n_neg}",
            "recall_value": round(tp / n_pos, 4),
            "false_escalation_rate": round(fp / n_neg, 4),
            "precision": round(tp / len(flagged), 4) if flagged else None,
        }
        key = (tp, fp, fn)
        if key in seen:
            # Same confusion row as a previous threshold: note it, keep the first row.
            rows[seen[key]]["also_at"].append(row["threshold"])
            if t in FIELD_REFERENCES[field]:
                rows[seen[key]]["is_reference"] = True
            continue
        seen[key] = len(rows)
        row["also_at"] = []
        row["is_reference"] = t in FIELD_REFERENCES[field]
        rows.append(row)
    return rows


def markdown_table(rows: list[dict], field: str) -> str:
    head = (
        f"| threshold on `{field}` | recall | false escalations | precision | notes |\n"
        "|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        notes = []
        label = FIELD_LABELS.get((field, r["threshold"]))
        if label:
            notes.append(label)
        if r["also_at"]:
            notes.append("same counts at " + ", ".join(f"{x:.4f}" for x in r["also_at"]))
        precision = "n/a" if r["precision"] is None else f"{r['precision']:.3f}"
        body.append(
            f"| {r['threshold']:.4f} | {r['recall']} | {r['false_escalations']} | "
            f"{precision} | {'; '.join(notes)} |"
        )
    return head + "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = json.loads(args.infile.read_text(encoding="utf-8"))
    cases = data["cases"]

    # --- integrity checks: fail loudly rather than print a number we cannot defend.
    if len(cases) != 22:
        raise SystemExit(f"expected 22 cases in the sweep, found {len(cases)}")
    if any(not c.get("ok", False) for c in cases):
        raise SystemExit("the sweep contains failed cases; re-run sweep_fixtures.py")
    n_pos = sum(1 for c in cases if c["label"] == "crisis")
    n_neg = len(cases) - n_pos
    if (n_pos, n_neg) != (8, 14):
        raise SystemExit(f"expected 8 crisis / 14 other, found {n_pos}/{n_neg}")

    labels = [1 if c["label"] == "crisis" else 0 for c in cases]

    report: dict = {
        "generated_with": "scripts/operating_points.py",
        "input": str(args.infile.relative_to(REPO)).replace("\\", "/"),
        "definitions": {
            "positive": "label == 'crisis'",
            "negative": "label in {'calm', 'check_in'}",
            "n_positive": n_pos,
            "n_negative": n_neg,
            "flag_rule": "score >= threshold",
            "recall": "TP / n_positive",
            "false_escalations": "FP / n_negative",
            "precision": "TP / (TP + FP)",
            "auc": "Mann-Whitney U via average ranks; ties count 1/2",
        },
        "sweeps": {},
    }

    for field in ("is_emergency", "escalating"):
        scores = [c[field] for c in cases]
        auc = auc_rank(scores, labels)
        auc_bf = auc_bruteforce(scores, labels)
        if abs(auc - auc_bf) > 1e-9:
            raise SystemExit(f"{field}: rank AUC {auc} != brute-force {auc_bf}")
        report["sweeps"][field] = {
            "auc": round(auc, 4),
            "auc_bruteforce_check": round(auc_bf, 4),
            "score_range": {"min": min(scores), "max": max(scores)},
            "rows": sweep(cases, field),
        }

    # Shipped OR rule, for a visible derivation of the benchmark's 6/8.
    flagged = [
        c
        for c in cases
        if c["is_emergency"] >= CURRENT_HIGH or c["escalating"] >= ESC_THRESHOLD
    ]
    tp = sum(1 for c in flagged if c["label"] == "crisis")
    report["shipped_or_rule"] = {
        "rule": f"is_emergency >= {CURRENT_HIGH} OR escalating >= {ESC_THRESHOLD}",
        "tp": tp,
        "fp": len(flagged) - tp,
        "recall": f"{tp}/{n_pos}",
        "false_escalations": f"{len(flagged) - tp}/{n_neg}",
    }

    # Clear-emergency subset: the separation claim in the docs rests on this. Report
    # both control sets, because which controls you pick changes the numbers.
    report["clear_emergency_subset"] = {}
    for name, controls in (("clear_controls", CLEAR_CONTROLS), ("plus_passive_ideation", CLEAR_CONTROLS | HARD_CONTROLS)):
        wanted = CLEAR_EMERGENCIES | controls
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise SystemExit(f"subset ids missing from the sweep: {sorted(missing)}")
        subset = [c for c in cases if c["id"] in wanted]
        sub_labels = [1 if c["label"] == "crisis" else 0 for c in subset]
        sub_scores = [c["is_emergency"] for c in subset]
        sub_auc = auc_rank(sub_scores, sub_labels)
        sub_bf = auc_bruteforce(sub_scores, sub_labels)
        if abs(sub_auc - sub_bf) > 1e-9:
            raise SystemExit("subset AUC mismatch")
        pos = [s for s, y in zip(sub_scores, sub_labels) if y == 1]
        neg = [s for s, y in zip(sub_scores, sub_labels) if y == 0]
        report["clear_emergency_subset"][name] = {
            "ids": sorted(wanted),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "auc": round(sub_auc, 4),
            "crisis_min": min(pos),
            "control_max": max(neg),
            "separates_at_this_subset": min(pos) > max(neg),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # --- stdout for pasting into the docs.
    print(f"input: {report['input']}  ({len(cases)} cases, {n_pos} crisis / {n_neg} other)")
    print("method: flag = score >= threshold; recall = TP/8; false escalations = FP/14\n")
    for field in ("is_emergency", "escalating"):
        s = report["sweeps"][field]
        print(f"### `{field}`  AUC = {s['auc']:.4f}  (brute-force check {s['auc_bruteforce_check']:.4f})")
        print(markdown_table(s["rows"], field))
        print()
    orr = report["shipped_or_rule"]
    print(f"### shipped OR rule\n{orr['rule']}\nrecall {orr['recall']}, false escalations {orr['false_escalations']}\n")
    print("### clear-emergency subset (6 emergencies vs controls)")
    for name, sub in report["clear_emergency_subset"].items():
        print(
            f"- {name}: n={sub['n_positive']} vs {sub['n_negative']}; AUC {sub['auc']:.4f}; "
            f"crisis min {sub['crisis_min']:.4f} vs control max {sub['control_max']:.4f}; "
            f"separates = {sub['separates_at_this_subset']}"
        )
    print()
    print(f"wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())