"""Tests for `scripts/evaluate_pipeline.py`. No network, no credentials.

The script is the only thing that runs the production triage path end to end over a
fixtures file, so these tests guard three separate things: that the arithmetic in the
summary is right, that the *production* OR rule is what actually runs (not a copy of it
living in the script), and that the guards around overwriting and held-out sets hold.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from crisis_triage.bedrock.safety_check import SafetyVerdict  # noqa: E402
from crisis_triage.decider_client import (  # noqa: E402
    DeciderError,
    DeciderResponse,
    NoulAnswer,
)
from crisis_triage.policies.crisis_policy import CrisisTriagePolicy  # noqa: E402
from crisis_triage.policies.questions import build_state  # noqa: E402

SCRIPT = REPO / "scripts" / "evaluate_pipeline.py"


def _load_script():
    """Load the script by path: `scripts/` is not a package and is not on sys.path."""
    spec = importlib.util.spec_from_file_location("evaluate_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script()

LOW, HIGH, ESC = 0.15, 0.30, 0.70


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class StateKeyedDecider:
    """Answers depend on the exact state string. Also proves the string is one string."""

    def __init__(self, table: dict[str, tuple[float, float]]) -> None:
        self.table = table
        self.calls: list[dict] = []

    def health(self, timeout_s: float = 10.0) -> dict:
        return {"status": "ok", "model": "state-keyed", "device": "cpu", "num_slots": 1}

    def evaluate(self, state: str, questions: dict, timeout_s=None) -> DeciderResponse:
        self.calls.append({"state": state, "questions": list(questions)})
        emergency, escalating = self.table[state]
        return DeciderResponse(
            answers={
                "is_emergency": NoulAnswer(probability=emergency),
                "escalating": NoulAnswer(probability=escalating),
            },
            model="state-keyed",
            latency_ms=111.0,
            input_tokens=50,
            output_tokens=2,
        )


class StateKeyedSafety:
    """Gate verdicts keyed by state, so the two legs can be compared on one string."""

    def __init__(self, gate: dict[str, bool] | None = None) -> None:
        self.gate = gate or {}
        self.calls: list[str] = []
        # The script reads model_id/region off the client; a double has no client.
        self.client = None

    def check(self, text: str) -> SafetyVerdict:
        self.calls.append(text)
        return SafetyVerdict(
            emergency=self.gate.get(text, False),
            reason="state-keyed",
            latency_ms=222.0,
            input_tokens=10,
            output_tokens=3,
        )


class ExplodingDecider:
    """Raises something the production fail-safe does NOT catch."""

    def health(self, timeout_s: float = 10.0) -> dict:
        return {"status": "ok", "model": "exploding", "device": "cpu", "num_slots": 1}

    def evaluate(self, state: str, questions: dict, timeout_s=None) -> DeciderResponse:
        raise RuntimeError("unexpected failure outside the fail-safe")


def write_cases(path: Path, cases: list[dict]) -> Path:
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return path


# Four crisis and four non-crisis cases whose routing is decided by hand, so every
# number in the summary can be asserted exactly. One turn each, so the state string is
# just the message and the table lookup is obvious.
ARITHMETIC_CASES = [
    {"id": "c_miss", "label": "crisis", "group": "clear_emergency", "turns": ["turn c1"]},
    {"id": "c_gate", "label": "crisis", "group": "clear_emergency", "turns": ["turn c2"]},
    {"id": "c_band", "label": "crisis", "group": "clear_emergency", "turns": ["turn c3"]},
    {"id": "c_esc", "label": "crisis", "group": "clear_emergency", "turns": ["turn c4"]},
    {"id": "n_ok", "label": "calm", "group": "clear_distress", "turns": ["turn n1"]},
    {"id": "n_band", "label": "calm", "group": "clear_distress", "turns": ["turn n2"]},
    {"id": "n_mid", "label": "check_in", "group": "edge_case", "turns": ["turn n3"]},
    {"id": "n_gate", "label": "check_in", "group": "edge_case", "turns": ["turn n4"]},
]

# state -> (P(is_emergency), P(escalating))
ARITHMETIC_TABLE = {
    "turn c1": (0.10, 0.20),  # calm band, quiet escalating, gate quiet  -> CALM   (a miss)
    "turn c2": (0.10, 0.20),  # calm band, quiet escalating, gate fires  -> CRISIS (gate only)
    "turn c3": (0.40, 0.20),  # band fires                                -> CRISIS (band only)
    "turn c4": (0.10, 0.80),  # escalating fires                          -> CRISIS (escalating only)
    "turn n1": (0.05, 0.10),  # everything quiet                          -> CALM
    "turn n2": (0.40, 0.10),  # band fires                                -> CRISIS (false escalation)
    "turn n3": (0.20, 0.10),  # mid band                                  -> CHECK_IN
    "turn n4": (0.20, 0.10),  # mid band, gate fires                      -> CRISIS (gate only)
}

ARITHMETIC_GATE = {"turn c2": True, "turn n4": True}


@pytest.fixture
def arithmetic(tmp_path):
    fixtures = write_cases(tmp_path / "arithmetic_cases.json", ARITHMETIC_CASES)
    out = tmp_path / "pipeline_arithmetic_cases.json"
    decider = StateKeyedDecider(ARITHMETIC_TABLE)
    safety = StateKeyedSafety(ARITHMETIC_GATE)
    report = mod.run(
        fixtures,
        out,
        decider=decider,
        safety=safety,
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=True,
    )
    return report, decider, safety, out


def _text(counts: dict | str) -> str:
    """The k/n text of a `_counts` row. Signal rows already store a plain "k/n" string."""
    return counts if isinstance(counts, str) else counts["text"]


# ---------------------------------------------------------------------------
# 1. metric arithmetic
# ---------------------------------------------------------------------------


def test_summary_arithmetic_is_exact(arithmetic):
    report, _, _, _ = arithmetic
    summary = report["summary"]
    signals = summary["signals"]

    assert summary["n_crisis_labelled"] == 4
    assert summary["n_non_crisis_labelled"] == 4

    # crisis recall: c_gate, c_band, c_esc. c_miss is routed to calm -> the dangerous failure.
    assert _text(summary["crisis_recall"]) == "3/4"
    assert summary["crisis_recall"]["ids"] == ["c_gate", "c_band", "c_esc"]
    assert _text(summary["emergencies_below_crisis"]) == "1/4"
    assert summary["emergencies_below_crisis"]["ids"] == ["c_miss"]
    assert _text(summary["false_escalations"]) == "2/4"
    assert summary["false_escalations"]["ids"] == ["n_band", "n_gate"]

    # strict agreement: c_gate, c_band, c_esc, n_ok, n_mid = 5.
    # lenient additionally accepts n_gate (check_in label, crisis route) = 6.
    assert _text(summary["correct_strict"]) == "5/8"
    assert _text(summary["correct_lenient"]) == "6/8"

    # each signal alone, over the same denominators
    assert _text(signals["safety_gate_alone"]["crisis_caught"]) == "1/4"
    assert _text(signals["safety_gate_alone"]["non_crisis_flagged"]) == "1/4"
    assert _text(signals["band_alone"]["crisis_caught"]) == "1/4"
    assert _text(signals["band_alone"]["non_crisis_flagged"]) == "1/4"
    assert _text(signals["escalating_alone"]["crisis_caught"]) == "1/4"
    assert _text(signals["escalating_alone"]["non_crisis_flagged"]) == "0/4"
    assert _text(signals["band_or_escalating"]["crisis_caught"]) == "2/4"
    assert _text(signals["band_or_escalating"]["non_crisis_flagged"]) == "1/4"
    assert _text(signals["final_pipeline"]["crisis_caught"]) == "3/4"
    assert _text(signals["final_pipeline"]["non_crisis_flagged"]) == "2/4"

    # the case only the gate catches: calm band, quiet escalating, gate fires
    gate_only = summary["crisis_caught_only_by_gate"]
    assert _text(gate_only["all_crisis"]) == "1/4"
    assert gate_only["all_crisis"]["ids"] == ["c_gate"]
    assert _text(gate_only["clear_emergencies"]) == "1/4"
    assert _text(summary["non_crisis_raised_only_by_gate"]) == "1/4"
    assert summary["non_crisis_raised_only_by_gate"]["ids"] == ["n_gate"]

    assert summary["n_errored"] == 0
    assert summary["n_state_mismatches"] == 0


def test_confusion_matrix_counts_every_case(arithmetic):
    report, _, _, _ = arithmetic
    counts = report["summary"]["confusion_label_x_route"]["counts"]
    assert counts["crisis"]["crisis"] == 3
    assert counts["crisis"]["calm"] == 1
    assert counts["calm"]["crisis"] == 1
    assert counts["calm"]["calm"] == 1
    assert counts["check_in"]["check_in"] == 1
    assert counts["check_in"]["crisis"] == 1
    total = sum(sum(row.values()) for row in counts.values())
    assert total == len(ARITHMETIC_CASES)


def test_every_metric_carries_a_definition(arithmetic):
    """Rule 4: a number must say what it means wherever it appears."""
    report, _, _, _ = arithmetic
    definitions = report["summary"]["definitions"]
    for key in (
        "crisis_recall",
        "emergencies_below_crisis",
        "false_escalations",
        "correct_strict",
        "correct_lenient",
        "denominators",
        "crisis_caught_only_by_gate",
    ):
        assert key in definitions
        assert definitions[key].strip()


# ---------------------------------------------------------------------------
# 2. the production OR path is what runs
# ---------------------------------------------------------------------------


def test_production_triage_state_is_what_runs(tmp_path, monkeypatch, fake_client, fake_safety):
    """The script must call the policy, not re-derive the route from the band."""
    cases = [
        {"id": "a", "label": "crisis", "group": "clear_emergency", "turns": ["one"]},
        {"id": "b", "label": "calm", "group": "clear_distress", "turns": ["two"]},
        {"id": "c", "label": "check_in", "group": "edge_case", "turns": ["three"]},
    ]
    fixtures = write_cases(tmp_path / "spy_cases.json", cases)
    out = tmp_path / "out.json"

    seen = []
    original = CrisisTriagePolicy.triage_state

    def spy(self, state, user_turns=None):
        outcome = original(self, state, user_turns)
        seen.append((state, list(user_turns or []), outcome))
        return outcome

    monkeypatch.setattr(CrisisTriagePolicy, "triage_state", spy)

    report = mod.run(
        fixtures,
        out,
        decider=fake_client,
        safety=fake_safety,
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=True,
    )

    assert len(seen) == len(cases)
    for (state, turns, outcome), row in zip(seen, report["cases"]):
        assert row["route"] == outcome.route.value
        assert row["triggered_by"] == outcome.triggered_by
        assert row["raised_by"] == "+".join(outcome.triggered_by)
        assert row["band_route"] == outcome.band.route.value
        assert row["state"] == state
        assert row["id"] in {case["id"] for case in cases}
        assert turns


def _one_case(tmp_path, name, turn):
    return write_cases(tmp_path / name, [
        {"id": "only", "label": "crisis", "group": "clear_emergency", "turns": [turn]}
    ])


def _run_one(tmp_path, name, turn, decider, safety, gate=None):
    safety = safety or StateKeyedSafety(gate or {})
    return mod.run(
        _one_case(tmp_path, name, turn),
        tmp_path / "out.json",
        decider=decider,
        safety=safety,
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=False,
    )


def test_or_rule_band_calm_but_gate_emergency(tmp_path):
    decider = StateKeyedDecider({"gate only": (0.05, 0.10)})
    report = _run_one(tmp_path, "or_gate.json", "gate only", decider, None, {"gate only": True})
    row = report["cases"][0]
    assert row["band_route"] == "calm"
    assert row["route"] == "crisis"
    assert row["triggered_by"] == ["bedrock_safety"]


def test_or_rule_band_crisis_only(tmp_path):
    decider = StateKeyedDecider({"band only": (0.90, 0.10)})
    report = _run_one(tmp_path, "or_band.json", "band only", decider, None)
    row = report["cases"][0]
    assert row["band_route"] == "crisis"
    assert row["route"] == "crisis"
    assert row["triggered_by"] == ["decider_high_band"]


def test_or_rule_escalating_only(tmp_path):
    decider = StateKeyedDecider({"esc only": (0.10, 0.95)})
    report = _run_one(tmp_path, "or_esc.json", "esc only", decider, None)
    row = report["cases"][0]
    assert row["band_route"] == "calm"
    assert row["route"] == "crisis"
    assert row["triggered_by"] == ["decider_escalating"]


# ---------------------------------------------------------------------------
# 3. one state string for both legs
# ---------------------------------------------------------------------------


def test_one_state_string_for_both_legs(tmp_path, fake_client, fake_safety):
    turn = "I have a plan and I bought what I need"
    fixtures = _one_case(tmp_path, "state_cases.json", turn)
    report = mod.run(
        fixtures,
        tmp_path / "out.json",
        decider=fake_client,
        safety=fake_safety,
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=True,
    )
    row = report["cases"][0]
    expected = build_state([turn], 6)

    assert row["state"] == expected
    assert row["state_matches_build_state"] is True
    assert row["state_same_for_both"] is True
    # Decider and the safety check both received that exact string.
    assert fake_client.calls[0]["state"] == expected
    assert fake_safety.calls[0] == expected


def test_multiturn_state_is_labelled_and_capped(tmp_path):
    turns = [f"message {i}" for i in range(1, 9)]
    decider = StateKeyedDecider({build_state(turns, 6): (0.05, 0.10)})
    safety = StateKeyedSafety()
    fixtures = write_cases(
        tmp_path / "multi_cases.json",
        [{"id": "m", "label": "calm", "group": "multi_turn", "turns": turns}],
    )
    report = mod.run(
        fixtures,
        tmp_path / "out.json",
        decider=decider,
        safety=safety,
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=False,
    )
    row = report["cases"][0]
    # The canonical builder keeps the last max_turns turns, labelled and oldest first.
    assert row["state"] == "\n".join(f"USER: message {i}" for i in range(3, 9))
    assert row["state"] == safety.calls[0]
    assert row["n_turns"] == 8


# ---------------------------------------------------------------------------
# 4. guards
# ---------------------------------------------------------------------------


def test_out_guard_requires_force(tmp_path):
    out = tmp_path / "out.json"
    assert mod.out_guard(out, force=False, heldout=False, confirm_heldout=False) is None
    out.write_text("{}", encoding="utf-8")
    reason = mod.out_guard(out, force=False, heldout=False, confirm_heldout=False)
    assert reason and "--force" in reason
    assert mod.out_guard(out, force=True, heldout=False, confirm_heldout=False) is None


def test_out_guard_heldout_is_evaluated_once(tmp_path):
    # `is_heldout` tests the WHOLE path on purpose, so the negative case uses a literal
    # path. A pytest `tmp_path` embeds this test's own name, which contains "heldout" --
    # every path under it would match, and the negative assertion would pass vacuously.
    assert mod.is_heldout(Path("tests/fixtures/heldout_cases.json")) is True
    assert mod.is_heldout(Path("tests/fixtures/heldout/test_cases.json")) is True
    assert mod.is_heldout(Path("tests/fixtures/test_cases.json")) is False
    assert mod.is_heldout(Path("HELDOUT_cases.json")) is True  # case-insensitive

    out = tmp_path / "pipeline_heldout_cases.json"

    reason = mod.out_guard(out, force=False, heldout=True, confirm_heldout=False)
    assert reason and "--confirm-heldout" in reason

    assert mod.out_guard(out, force=False, heldout=True, confirm_heldout=True) is None

    out.write_text("{}", encoding="utf-8")
    reason = mod.out_guard(out, force=True, heldout=True, confirm_heldout=True)
    assert reason and "once" in reason


def _patch_factories(monkeypatch, decider, safety):
    monkeypatch.setattr(mod, "make_decider", lambda url, timeout_s: decider)
    monkeypatch.setattr(mod, "make_safety", lambda: safety)
    monkeypatch.setenv("CRISIS_LOW", str(LOW))
    monkeypatch.setenv("CRISIS_HIGH", str(HIGH))
    monkeypatch.setenv("CRISIS_ESC", str(ESC))


def _quiet_fakes():
    return (
        StateKeyedDecider({"hello": (0.05, 0.10)}),
        StateKeyedSafety(),
    )


def test_main_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    decider, safety = _quiet_fakes()
    _patch_factories(monkeypatch, decider, safety)
    fixtures = write_cases(
        tmp_path / "cli_cases.json",
        [{"id": "x", "label": "calm", "group": "clear_distress", "turns": ["hello"]}],
    )
    out = tmp_path / "written.json"

    assert mod.main(["--fixtures", str(fixtures), "--out", str(out)]) == 0
    assert out.exists()

    # second run without --force is refused, and the first file is left alone
    before = out.read_text(encoding="utf-8")
    assert mod.main(["--fixtures", str(fixtures), "--out", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == before

    assert mod.main(["--fixtures", str(fixtures), "--out", str(out), "--force"]) == 0


def test_main_requires_confirm_heldout(tmp_path, monkeypatch):
    decider, safety = _quiet_fakes()
    _patch_factories(monkeypatch, decider, safety)
    fixtures = write_cases(
        tmp_path / "heldout_cases.json",
        [{"id": "x", "label": "calm", "group": "clear_distress", "turns": ["hello"]}],
    )
    out = tmp_path / "pipeline_heldout_cases.json"

    # no --confirm-heldout: refused, nothing written
    assert mod.main(["--fixtures", str(fixtures), "--out", str(out)]) == 2
    assert not out.exists()

    # --confirm-heldout but the output already exists: still refused, even with --force
    out.write_text("{}", encoding="utf-8")
    args = ["--fixtures", str(fixtures), "--out", str(out), "--confirm-heldout"]
    assert mod.main(args) == 2
    assert mod.main([*args, "--force"]) == 2
    assert out.read_text(encoding="utf-8") == "{}"

    # the one legitimate run: confirmed, and nothing there yet
    out.unlink()
    assert mod.main(args) == 0
    assert out.exists()


def test_missing_fixtures_file_is_reported(tmp_path, monkeypatch):
    decider, safety = _quiet_fakes()
    _patch_factories(monkeypatch, decider, safety)
    assert mod.main(["--fixtures", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.json")]) == 2


# ---------------------------------------------------------------------------
# 5. error handling
# ---------------------------------------------------------------------------


def test_decider_error_fails_safe_and_is_counted(tmp_path):
    """A dead Decider is production's fail-safe, not a script error: route CRISIS."""
    cases = [
        {"id": "one", "label": "crisis", "group": "clear_emergency", "turns": ["a"]},
        {"id": "two", "label": "calm", "group": "clear_distress", "turns": ["b"]},
    ]
    fixtures = write_cases(tmp_path / "dead_cases.json", cases)
    decider = StateKeyedDecider({})
    decider.evaluate = lambda state, questions, timeout_s=None: (_ for _ in ()).throw(
        DeciderError("connection refused")
    )
    safety = StateKeyedSafety()

    report = mod.run(
        fixtures,
        tmp_path / "out.json",
        decider=decider,
        safety=safety,
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=True,
    )

    summary = report["summary"]
    assert summary["n_errored"] == 0  # the fail-safe handled it; nothing was dropped
    assert summary["n_fail_safe"] == 2
    # One crisis case, one calm case in this file: the dead Decider sends the crisis case
    # to CRISIS via the fail-safe (recall 1/1) and also escalates the calm case (1/1).
    assert summary["crisis_recall"]["text"] == "1/1"
    assert summary["false_escalations"]["text"] == "1/1"
    for row in report["cases"]:
        assert row["route"] == "crisis"
        assert row["fallback_reason"]
        assert row["triggered_by"] == ["decider_unavailable", "decider_high_band"]
        assert row["p_emergency"] is None  # NaN is written as null, never as NaN

    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert len(written["cases"]) == 2
    assert "NaN" not in (tmp_path / "out.json").read_text(encoding="utf-8")


def test_unexpected_exception_is_recorded_and_counted(tmp_path):
    """Anything the fail-safe does not catch is reported, counted, and never dropped."""
    cases = [
        {"id": "one", "label": "crisis", "group": "clear_emergency", "turns": ["a"]},
        {"id": "two", "label": "calm", "group": "clear_distress", "turns": ["b"]},
    ]
    fixtures = write_cases(tmp_path / "boom_cases.json", cases)
    report = mod.run(
        fixtures,
        tmp_path / "out.json",
        decider=ExplodingDecider(),
        safety=StateKeyedSafety(),
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=True,
    )
    summary = report["summary"]
    assert summary["n_errored"] == 2
    assert summary["errored_ids"] == ["one", "two"]
    # An errored case still counts in the denominators, in the conservative direction.
    # This file has one crisis case and one calm case, and neither produced a route.
    assert summary["crisis_recall"]["text"] == "0/1"
    assert summary["false_escalations"]["text"] == "0/1"
    for row in report["cases"]:
        assert row["route"] is None
        assert "RuntimeError" in row["error"]
        assert row["state"] is not None

    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert len(written["cases"]) == len(cases)
    assert written["summary"]["n_errored"] == 2


def test_errored_case_counts_against_the_crisis_denominator(tmp_path):
    """The conservative direction: no route means the case is a crisis miss, not excluded."""
    cases = [
        {"id": "c1", "label": "crisis", "group": "clear_emergency", "turns": ["a"]},
        {"id": "c2", "label": "crisis", "group": "clear_emergency", "turns": ["b"]},
    ]
    fixtures = write_cases(tmp_path / "mixed_cases.json", cases)

    class OneGoodOneBad:
        def health(self, timeout_s: float = 10.0) -> dict:
            return {"status": "ok", "model": "mixed", "device": "cpu", "num_slots": 1}

        def evaluate(self, state: str, questions: dict, timeout_s=None) -> DeciderResponse:
            if state == "b":
                raise RuntimeError("boom")
            return DeciderResponse(
                answers={
                    "is_emergency": NoulAnswer(probability=0.90),
                    "escalating": NoulAnswer(probability=0.10),
                },
                model="mixed",
                latency_ms=100.0,
                input_tokens=10,
                output_tokens=2,
            )

    report = mod.run(
        fixtures,
        tmp_path / "out.json",
        decider=OneGoodOneBad(),
        safety=StateKeyedSafety(),
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=False,
    )
    summary = report["summary"]
    assert summary["crisis_recall"]["text"] == "1/2"
    assert summary["emergencies_below_crisis"]["ids"] == ["c2"]
    assert summary["n_errored"] == 1
    assert "c2" in summary["errored_ids"]


# ---------------------------------------------------------------------------
# 6. no secrets in the output
# ---------------------------------------------------------------------------


def test_no_credentials_are_written_or_printed(tmp_path, monkeypatch, capsys):
    sentinel = "SENTINEL-DO-NOT-LOG-9f8e7d6c"
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", sentinel)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", sentinel)

    decider = StateKeyedDecider({"hello": (0.05, 0.10)})
    fixtures = write_cases(
        tmp_path / "secret_cases.json",
        [{"id": "s", "label": "calm", "group": "clear_distress", "turns": ["hello"]}],
    )
    report = mod.run(
        fixtures,
        tmp_path / "out.json",
        decider=decider,
        safety=StateKeyedSafety(),
        low=LOW,
        high=HIGH,
        esc_threshold=ESC,
        warmup=False,
        write=True,
    )
    captured = capsys.readouterr()

    blob = json.dumps(report)
    assert sentinel not in blob
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert sentinel not in (tmp_path / "out.json").read_text(encoding="utf-8")
    # The key is reported only as a boolean.
    assert report["header"]["bedrock"]["api_key_present"] is True
    # Printed only as a boolean, and only as the key's presence.
    assert "key_present=True" in captured.out


# ---------------------------------------------------------------------------
# 7. wiring, and the dev-set note
# ---------------------------------------------------------------------------


def test_main_writes_the_report_and_flags_the_dev_set(tmp_path, monkeypatch, capsys):
    decider = StateKeyedDecider({"hello": (0.05, 0.10)})
    _patch_factories(monkeypatch, decider, StateKeyedSafety())

    dev = write_cases(
        tmp_path / "test_cases.json",
        [{"id": "d", "label": "calm", "group": "clear_distress", "turns": ["hello"]}],
    )
    out = tmp_path / "pipeline_test_cases.json"
    assert mod.main(["--fixtures", str(dev), "--out", str(out)]) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["header"]["dev_set_note"]
    assert "optimistic" in written["header"]["dev_set_note"]
    assert "optimistic" in capsys.readouterr().out

    other = write_cases(
        tmp_path / "other_cases.json",
        [{"id": "o", "label": "calm", "group": "clear_distress", "turns": ["hello"]}],
    )
    out2 = tmp_path / "pipeline_other_cases.json"
    assert mod.main(["--fixtures", str(other), "--out", str(out2)]) == 0
    written2 = json.loads(out2.read_text(encoding="utf-8"))
    assert written2["header"]["dev_set_note"] is None


def test_header_records_what_a_rerun_needs(tmp_path, monkeypatch):
    decider = StateKeyedDecider({"hello": (0.05, 0.10)})
    _patch_factories(monkeypatch, decider, StateKeyedSafety())
    fixtures = write_cases(
        tmp_path / "header_cases.json",
        [{"id": "h", "label": "calm", "group": "clear_distress", "turns": ["hello"]}],
    )
    out = tmp_path / "out.json"
    assert mod.main(["--fixtures", str(fixtures), "--out", str(out)]) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    header = written["header"]
    for key in (
        "git_commit",
        "git_dirty",
        "timestamp_utc",
        "fixtures_sha256",
        "thresholds",
        "bedrock",
        "decider_health",
        "questions",
        "n_cases",
        "n_errored",
    ):
        assert key in header, key

    assert header["thresholds"]["low"] == LOW
    assert header["thresholds"]["high"] == HIGH
    assert header["thresholds"]["escalating"] == ESC
    assert header["thresholds"]["low_source"] == "environment:CRISIS_LOW"
    assert header["decider_health"]["ok"] is True
    assert header["decider_health"]["body"]["model"] == "state-keyed"
    assert set(header["questions"]["names"]) >= {"is_emergency", "escalating"}
    assert header["questions"]["sha256"]
    # sha256 of the exact bytes the script read
    import hashlib

    assert header["fixtures_sha256"] == hashlib.sha256(fixtures.read_bytes()).hexdigest()
