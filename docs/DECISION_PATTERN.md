# Decision pattern: confidence banding with a local decision model

**Pattern tag:** `#policy` · **Model:** Strands Decider 2B · **Written against** the final code.

A local decision model scores a typed yes/no question on every turn. Two thresholds, read off
your own labelled data, turn that probability into a route, and the route selects how the agent
behaves. The pattern is domain-agnostic — crisis triage is one configuration of it.

## The decision

| | |
|---|---|
| **Question** | One `noul` (yes/no) question, e.g. *"Does the latest message indicate a genuine emergency requiring immediate crisis resources?"* |
| **Options** | `true` / `false`, optionally with written `criteria` descriptions (the server supplies generic defaults when you omit them) |
| **Output** | The model's `P(true)` — not a token, not a label choice |
| **Decision** | Compare `P(true)` to `low` and `high`, then route |
| **Extras** | A second `noul` ("is it escalating?") and a `choice`/`score` pair for context |

**On "calibrated":** the model is calibrated on AWS's own benchmark. We have **not** validated
that its probabilities are calibrated on our domain, so this document says "the model's score"
rather than "a calibrated probability". Treat the number as a score whose *ordering* you have
checked, not as a probability you can spend.

**On `criteria`:** an early third-party report said that omitting `noul` criteria clusters every
answer between 0.3 and 0.65. The shipped server already substitutes default criteria, and we have
not run the ablation that would isolate their effect here — so "descriptions matter" is an open
question for us, not a claim. Write them because they clarify the boundary you mean; measure
before asserting they change the scores.

## Why bands instead of one cutoff

A single cutoff throws away the information that the probability is *uncertain*. A middle band
is the model saying "I cannot tell" — which is a real, actionable answer: ask one more
question and re-score on the next turn. In a crisis context that is the `check_in` route: one
gentle, direct question with the support line still visible.

```
P >= high        -> CRISIS    decisive resources, one concrete next step
low < P < high   -> CHECK_IN  one clarifying question; re-score next turn
P <= low         -> CALM      listen, offer options, no directives
```

## Measure before you set the thresholds

This is the part that must not be skipped, and it is where our first attempt failed.

```python
# scripts/sweep_fixtures.py — before writing a policy
# 1. Run your labelled fixtures through the model.
# 2. Print min/median/max of P by label.
# 3. Check whether a threshold can separate the classes at all.
```

Our measured distribution (22 fixtures):

| label | min | median | max |
|---|---|---|---|
| calm | 0.0791 | 0.0877 | 0.1077 |
| check-in | 0.0971 | 0.1275 | 0.1765 |
| crisis | 0.1416 | 0.2616 | 0.8551 |

`crisis_min (0.1416) < non_crisis_max (0.1765)`: **the tails overlap**, so no single threshold
gets every case right. That is narrower than it sounds, and the shape matters:

- The **six clear emergencies do separate** from the six clear controls at our wording
  (lowest emergency 0.177 vs highest control 0.131).
- The overlap comes from the **ambiguous and multi-turn** cases, where a check-in case scores
  above two of the eight emergencies.
- A **low cutoff near 0.1416** catches all eight at the cost of some false escalations — a
  defensible safety-biased trade, not a bug.

Rather than argue "separates / doesn't", read the **operating-point table** in
[`docs/GROUND_TRUTH.md` §4b](GROUND_TRUTH.md): recall and false escalations at every observed
threshold, with AUC, all recomputed from the saved sweep. Pick your cutoff from that table
according to what a false negative costs you.

The documented default of `HIGH = 0.75` is a separate failure: it sits above **every** case but
one, so it would have routed 7 of 8 emergencies to `CALM`. That number was a guess about a
scale, not a considered bar.

Two lessons worth more than the pattern itself:

1. **The model's output scale is not the scale you assumed.** Never write `HIGH = 0.75` because
   it looks like a sensible confidence bar. Read it off your own data.
2. **Rewording may not be the fix.** We scored four alternative formulations (three rewrites and
   a `score` form) and none separated better than the original. That does not prove wording can
   never help — it does mean we stopped tuning prompts and changed the architecture instead of
   burning the weekend on them.

Because no threshold is perfect, do not rely on the band for the decision that must not be
wrong. Keep the band as the **tone** decision and add an independent **safety** gate.

## Safety is not the same decision as tone

Tone and safety have different tolerances for error. Tone can be wrong occasionally; safety
cannot. Splitting them lets each be owned by the component that is actually good at it, and
the safety gate can be a conservative OR rather than a single number:

```
route = CRISIS if any of:
    safety_gate(text)            # an independent, stronger classifier
    probability >= high          # the band says high
    escalating >= esc_threshold  # the conversation is getting worse
    model_unavailable            # fail safe, never silence
```

In our run this rescued **two of six clear emergencies** that the local model scored below
every non-crisis case, and gave **8/8 crisis recall with zero emergencies below crisis** — on the
22 fixtures that were also used to choose the thresholds and shape the safety prompt, so treat
that as an optimistic number until a held-out set says otherwise.

### What it actually costs, and when the gate runs

State this plainly, because it is easy to design around the wrong mental model:

- `CrisisTriagePolicy.triage_state` calls `SafetyChecker.check` on **every turn** whenever a
  checker is configured. The gate is not a rare second opinion; it is part of the loop, and it
  costs one Bedrock call per turn. Decider does **not** save latency or cost on the safety
  decision.
- Measured on 22 fixtures: the local model is **~335 ms p50** (2 questions; a quieter sweep
  measured 165 ms — see GROUND_TRUTH §3), and the **pipeline safety gate is 1012 ms p50 /
  1824 ms p95**. The 780 ms p50 figure sometimes quoted for Bedrock is the *benchmark
  classifier*, which uses a much shorter prompt than the safety gate.
- What the local model buys is a cheap, local, per-turn **tone and escalation** signal that
  decides how the reply is framed — not a way to skip the gate. A design that used it as a
  pre-filter to skip the safety call exists only as an untested hypothesis in our notes; we did
  not adopt it, because skipping a safety check on weak evidence is the wrong direction here.

## API

```python
from crisis_triage.policies.confidence_band import ConfidenceBandPolicy, Route

policy = ConfidenceBandPolicy(
    client,                  # anything with .evaluate(state, questions) -> DeciderResponse
    question="is_emergency", # must name a `noul` question in the question set
    low=0.15,
    high=0.30,               # omit both to read CRISIS_LOW/CRISIS_HIGH from the environment
)

decision = policy.decide(user_turns)
decision.route                   # Route.CRISIS | Route.CHECK_IN | Route.CALM
decision.probability             # the model's score for P(true); ordering checked, calibration not
decision.escalating_probability  # reported, not acted on
decision.explain()               # one audit line
```

Three deliberate constraints:

- **Thresholds have no defaults.** `BandPolicyError` if unset. A plausible default is how a
  wrong band ships.
- **A Decider failure raises.** `decide_or_fallback()` exists for callers that want the
  crisis-domain answer (route CRISIS, mark `fallback_reason`), but that choice is the caller's,
  not the pattern's.
- **Escalation is reported, not applied.** It is a weak signal; the caller ORs it in.

## Wiring into Strands

```python
from strands import Agent
from strands.models import BedrockModel
from crisis_triage.policies.crisis_policy import CrisisTriagePolicy, crisis_intervention_handler

triage = CrisisTriagePolicy(band_policy, SafetyChecker())
agent  = Agent(
    model=BedrockModel(),
    system_prompt="...",
    interventions=[crisis_intervention_handler(triage)],
)
```

The handler hangs off two hooks, and both choices are load-bearing:

- **`before_invocation`** — triage once per turn. Not `before_agent_call`; that hook does not
  exist. Capture the decision in `event.invocation_state`.
- **`before_model_call`** — inject the directive with `Guide(feedback=...)` and clear the stored
  decision so it is injected exactly once. `Guide` returned from `before_invocation` *cancels*
  the invocation instead of guiding it; `Guide` from `before_model_call` appends the feedback as
  a user message and lets the model proceed.

Handlers must define a `name` property, and **only class-level overrides are detected** —
assigning a method on the instance is silently ignored.

## Adapting to another domain

Replace the question and the routes; keep the shape.

```python
# Support inbox: is this message abusive?
ABUSE = {"type": "noul",
         "instructions": "Does this message contain abuse directed at staff?",
         "criteria": {"true": "Insults, threats, or slurs aimed at a person",
                      "false": "Angry or blunt, but no abuse aimed at a person"}}
policy = ConfidenceBandPolicy(client, question="is_abuse", low=0.20, high=0.55,
                              questions={"is_abuse": ABUSE})
```

Then: sweep your own fixtures, plot P by label, and only then choose `low`/`high`. If the
classes overlap, keep the band as a tone signal and add an independent gate for the part that
must not be wrong.

## Files

| File | Role |
|---|---|
| `src/crisis_triage/policies/confidence_band.py` | the pattern |
| `src/crisis_triage/policies/questions.py` | the typed questions + state building |
| `src/crisis_triage/policies/crisis_policy.py` | crisis config, the safety OR, the handler |
| `src/crisis_triage/bedrock/safety_check.py` | the independent safety gate |
| `scripts/sweep_fixtures.py` | measures the distribution you must tune against |
| `examples/strands_agent_demo.py` | the pattern inside a real `Agent` |
| `docs/GROUND_TRUTH.md` | every measured number, with its command |