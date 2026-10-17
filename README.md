# Crisis Triage Agent

`#strands-decider-2b` `#policy`

A triage agent that runs a small local decision model on **every user turn** and uses those
calibrated probabilities to choose *how* to respond, while **Amazon Bedrock** decides whether
the person is safe and writes the reply.

**This is a triage and routing demo. It is not a diagnosis tool and not a substitute for
emergency or professional services.** If you or someone else is in danger, contact local
emergency services.

---

## The honest version of this project

Most entries will say "we gave Decider a decision and it worked." This one says what we
measured, including the part that did not work.

We asked **Strands Decider 2B** the one question that matters: *is this person in an
emergency?* Across 22 scripted fixtures it could not answer that reliably:

| intended label | n | min P(emergency) | median | max |
|---|---|---|---|---|
| calm | 9 | 0.0791 | 0.0877 | 0.1077 |
| check-in | 5 | 0.0971 | 0.1275 | 0.1765 |
| **crisis** | 8 | **0.1416** | 0.2616 | 0.8551 |

The **ranking** is right and monotone. The **scale** is compressed, and the classes overlap:
the lowest crisis case (0.1416) sits *below* the highest non-crisis case (0.1765). A stated
suicide plan with the means acquired scored **0.2158**. At the documented `HIGH = 0.75`, **7 of
8 emergencies fell through the crisis band.**

We then tried to fix it by rewording, since wording is the main tuning lever for this model.
Four formulations, none of which separated better than the original:

| question | separates? | crisis min | non-crisis max |
|---|---|---|---|
| current wording | yes (subset) | 0.177 | 0.131 |
| "immediate physical danger right now?" | no | 0.099 | 0.128 |
| "should a dispatcher send 911?" | no | 0.128 | 0.257 |
| `score`: immediate/imminent mass | no | 0.416 | 0.470 |

So we changed the architecture instead of the prompt.

## What the agent does

```
user turn
  ├─ Decider 2B (local GPU, p50 159-335 ms)  ->  P(emergency), P(escalating)  [tone + escalation]
  ├─ Bedrock safety check (p50 1064 ms)      ->  needs crisis resources now?  [safety]
  └─ OR of four signals -> crisis | check_in | calm -> Bedrock writes the reply
```

**Safety is a four-signal OR, and no emergency depends on Decider alone:**

```python
safety.emergency                 # Bedrock classifier
or band_route is CRISIS          # Decider's high band
or escalating >= CRISIS_ESC       # Decider's escalation signal
or decider_unreachable           # fail safe -- never silence
```

This is not belt-and-braces for its own sake. On the fixture set, **two of the six clear
emergencies were caught by the Bedrock check alone** — Decider scored the person standing on
a bridge at 0.1770 and a third party relaying a goodbye message with an address at 0.1804.

Decider's job is the thing it is genuinely good at here: a **331 ms p50** local read on tone
and escalation on every single turn, which is the creative core of the entry (the "check-in"
band and mid-conversation escalation watching).

## Measured results

All numbers from `results/benchmark.json` and `results/fixture_run.json`. Hardware: **Windows 11,
RTX 4060 Ti 16 GB (CUDA), Python 3.13.5, torch 2.11.0+cu128.** Decider served locally.

**Routing (22 fixtures, full pipeline):**

| metric | result |
|---|---|
| crisis recall | **8/8** |
| emergencies routed below crisis | **0** |
| overall band agreement | 19/22 |

**Decider vs Bedrock on the same classification task** (`scripts/benchmark.py`):

| | Decider 2B (local) | Bedrock LLM |
|---|---|---|
| latency p50 | **160 ms** | 846 ms |
| latency p95 | **170 ms** | 1011 ms |
| correct | 19/22 | **22/22** |
| crisis recall | 6/8 | **8/8** |
| API cost | $0 (runs on our GPU) | see below |

Agreement between the two: **86.4% (19/22)**.

The **safety check** used in the pipeline is a longer prompt than the benchmark's bare
classifier, so it is slower: **p50 1064 ms, p95 1813 ms** over the 22 fixtures (256 input /
46 output tokens per call). The benchmark above is the like-for-like comparison; the safety
check's own figures are what the running agent actually pays.

Token usage for the Bedrock classifier: **2254 input / 286 output** for 22 calls (~102 in /
13 out per call). **Cost is deliberately not stated** — `scripts/benchmark.py` reports `null`
unless you pass `--input-price`/`--output-price` from the current Bedrock pricing page. We did
not want a price in the README that we had not read off a live page.

Decider is **~2.3x faster at p50** and **measurably worse at the thing that must not be wrong.**
That trade is the reason for the split, and it is the honest version of "effective use of
Decider 2B."

## The reusable pattern: `#policy`

The submission asset is domain-agnostic. A typed yes/no question plus two calibrated
thresholds, and a route:

```python
from crisis_triage.policies.confidence_band import ConfidenceBandPolicy

policy = ConfidenceBandPolicy(client, question="is_fraudulent", low=0.15, high=0.30)
decision = policy.decide(user_turns)
decision.route          # CRISIS | CHECK_IN | CALM  (or your own band labels)
decision.probability    # the calibrated P
decision.escalating_probability
decision.explain()      # one audit line
```

Three deliberate design choices, each learned the hard way:

1. **It refuses to default its thresholds.** `ConfidenceBandPolicy` raises unless `low`/`high`
   are supplied or set in the environment. The values must come from a measured sweep; a
   plausible-looking default is how the 0.75/0.25 guess shipped in the first place.
2. **It does not fail safe.** On a Decider error it raises. The *caller* decides what an
   unreachable model means — in a crisis domain that answer is "show resources", and hiding
   that decision inside a generic policy would make it invisible.
3. **It reports escalation but does not act on it.** Escalation is a weak signal; the caller
   combines it. The full pattern is documented in [`docs/DECISION_PATTERN.md`](docs/DECISION_PATTERN.md).

Wiring it into a real Strands agent (see `examples/strands_agent_demo.py`):

```python
triage  = CrisisTriagePolicy(ConfidenceBandPolicy(DeciderClient(), low=0.15, high=0.30), SafetyChecker())
agent   = Agent(model=BedrockModel(), system_prompt=..., interventions=[crisis_intervention_handler(triage)])
```

Triage runs on `before_invocation`; the directive is injected on `before_model_call` via
`Guide(feedback=...)`. That split is required, not stylistic: `Guide` returned from
`before_invocation` *cancels* the invocation instead of guiding it.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# CUDA torch FIRST -- a plain `pip install torch` on Windows is CPU-only
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # must print True
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Copy `.env.example` to `.env` and fill in `AWS_BEARER_TOKEN_BEDROCK`. Then, in its own terminal:

```powershell
.\.venv\Scripts\strands-decider.exe serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8000
```

And in another:

```powershell
.\.venv\Scripts\python.exe -m crisis_triage.main --health     # both backends
.\.venv\Scripts\python.exe -m crisis_triage.main --test       # fixtures + crisis recall
.\.venv\Scripts\python.exe -m crisis_triage.main --demo       # scripted calm -> check-in -> crisis
.\.venv\Scripts\python.exe -m crisis_triage.main --interactive
.\.venv\Scripts\python.exe -m pytest tests -v                 # no server needed
```

**Bedrock model ID:** the plain `anthropic.claude-haiku-4-5-20251001-v1:0` is **rejected** for
on-demand throughput — it must go through an inference profile. `.env` pins
`us.anthropic.claude-haiku-4-5-20251001-v1:0`, verified working in both boto3 `converse` and
the Strands `BedrockModel` wrapper.

## Safety behaviour

- **Nothing fails silently.** An unreachable Decider routes to crisis resources, not to calm.
- **The Bedrock safety check fails closed.** If it cannot answer, the person is treated as
  needing resources.
- **Every path keeps a support line visible**, including the calm path. No path dismisses the
  user or withholds a number.
- **Resources are verified, never recalled:** 988 (call or text), chat at `chat.988lifeline.org`,
  Spanish available — read from <https://988lifeline.org/> on 2026-10-17. `911` for immediate
  danger. US-centric; that is a limitation, not a hidden assumption.
- **No user text is logged.** Demos and fixtures are scripted, non-personal examples.

## Limitations

- **Decider 2B is not reliable enough to route crises**, as measured above. We say so rather
  than quoting a figure that flatters it. This is a triage demo, not a safety system.
- Thresholds (`CRISIS_HIGH=0.30`, `CRISIS_LOW=0.15`, `CRISIS_ESC=0.70`) control **tone only**.
  They are not a safety boundary; the OR is.
- Cost figures are intentionally absent until a live pricing page is read.
- The safety check adds **p50 1064 ms / p95 1813 ms**, and the turn's total is that plus
  Decider's ~331 ms and the reply generation. That latency is the price of not letting a
  compressed probability decide whether someone gets help; a production system would run the
  gate concurrently with the tone decision.
- Fixtures are 22 scripted cases, not a clinical evaluation. Edge cases (passive ideation, past
  self-harm with no current intent, future fear) are included deliberately, and Decider's
  behaviour on them is in `results/fixtures_decider.json`.

## Layout

```
src/crisis_triage/
  main.py                    orchestrator + CLI (CrisisTriageAgent, TriageResult)
  decider_client.py          DeciderClient / DeciderResponse
  resources.py               verified crisis resources
  policies/
    questions.py             the typed questions + state building
    confidence_band.py       ConfidenceBandPolicy -- the reusable pattern
    crisis_policy.py         crisis config + the Strands InterventionHandler
  bedrock/
    client.py                shared Bedrock access
    safety_check.py          the safety classifier (fails closed)
    responders.py            crisis / check-in / calm reply styles
tests/                       mock-based; pass with no server and no credentials
scripts/
  sweep_fixtures.py          measures Decider across the fixtures
  benchmark.py               Decider vs Bedrock, latency + tokens
  smoke_bedrock.py           settles the model ID
examples/strands_agent_demo.py   the pattern inside a real Strands Agent
docs/GROUND_TRUTH.md         every measured number, with the command that produced it
docs/DECISION_PATTERN.md     how to reuse the pattern
```

## License

Apache-2.0. See `LICENSE`.