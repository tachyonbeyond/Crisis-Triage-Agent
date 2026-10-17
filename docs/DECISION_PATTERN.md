# Decision pattern: confidence banding with a local decision model

**Pattern tag:** `#policy` · **Model:** Strands Decider 2B · **Written against** the final code.

A local decision model scores a typed yes/no question on every turn. Two calibrated
thresholds turn that probability into a route, and the route selects how the agent behaves.
The pattern is domain-agnostic — crisis triage is one configuration of it.

## The decision

| | |
|---|---|
| **Question** | One `noul` (yes/no) question, e.g. *"Does the latest message indicate a genuine emergency requiring immediate crisis resources?"* |
| **Options** | `true` / `false`, each with a written `criteria` description. Descriptions matter: they are what make the model's boundary stable. |
| **Output** | A calibrated `P(true)` — not a token, not a label choice |
| **Decision** | Compare `P(true)` to `low` and `high`, then route |
| **Extras** | A second `noul` ("is it escalating?") and a `choice`/`score` pair for context |

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

`crisis_min (0.1416) < non_crisis_max (0.1765)`: **no threshold separates these classes.**
The documented default of `HIGH = 0.75` would have routed 7 of 8 emergencies to `CALM`.

Two lessons worth more than the pattern itself:

1. **The model's output scale is not the scale you assumed.** Never write `HIGH = 0.75` because
   it looks like a sensible confidence bar. Read it off your own fixtures.
2. **Rewording may not be the fix.** Wording is usually the tuning lever, but we scored four
   alternative formulations and none separated better — the problem was calibration, not
   phrasing. Knowing that changed the architecture instead of burning the weekend on prompts.

When separation fails, do not lower the threshold until emergencies pass and everything else
does too. Keep the band as the **tone** decision and add an independent **safety** gate.

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
every non-crisis case, and gave **8/8 crisis recall with zero emergencies below crisis**.

Cheap signals come first. The local model is ~165 ms (tone, every turn); the safety gate is
~780 ms p50. That is why the local model runs on every turn and the gate is a second opinion
rather than the whole loop.

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
decision.probability             # the calibrated P
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