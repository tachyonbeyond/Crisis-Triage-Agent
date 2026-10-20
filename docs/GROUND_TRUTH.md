# Ground truth — measured 2026-10-17

Every number here came from a command run on this machine. Nothing is estimated. If a claim in the README or the
article is not backed by this file or by `results/`, it does not belong in them.

Hardware / environment for all Decider numbers: **Windows 11, RTX 4060 Ti 16 GB, CUDA, Python 3.13.5, torch 2.11.0+cu128.**
Decider model: `StrandsAgents/strands-decider-2B-hobson-v19` (base `Qwen/Qwen3.5-2B-Base`), served locally on port 8000.

## 1. Environment facts

Two things were uncertain when this project started and are settled here: whether CUDA PyTorch
works on this machine at all, and whether the Decider weights needed a fresh download.

```
torch 2.11.0+cu128   cuda_available True   device NVIDIA GeForce RTX 4060 Ti
capability (8, 9)    vram_gb 16.0          bf16 supported True
```

- The venv is **Python 3.13.5**. CUDA PyTorch support for Python 3.14 was unverified, so this
  machine deliberately uses 3.13; no rebuild was needed.
- Weights were **already cached**: `models--Qwen--Qwen3.5-2B-Base` (4.3 GB) + `models--StrandsAgents--...-hobson-v19`
  (88 MB LoRA + head). No 4.6 GB download occurred.
- The Decider release ships as a **LoRA adapter + a separate readout head** on a Qwen3.5-2B base, not a standalone 2B
  checkpoint. `hobson_config.json` records `num_slots: 24`, `head_type: "pointer"`, `max_length: 4096`, `use_lora: true`.

## 2. Decider HTTP contract — verified live

`GET /health` **exists**. Live `openapi.json` exposes **only** these two routes: `/health` and
`/v1/systemone`.

```console
$ curl.exe -s http://127.0.0.1:8000/health
{"status":"ok","model":"strands-decider-2B-hobson-v19","checkpoint":"StrandsAgents/strands-decider-2B-hobson-v19",
 "base_model":"Qwen/Qwen3.5-2B-Base","num_slots":24,"max_length":4096,"temperature":0.9627721607677362,
 "device":"cuda","prefix_cache":true}
[http_code=200 time=0.002025s]
```

```console
$ curl.exe -s -X POST http://127.0.0.1:8000/v1/systemone \
    -H "Content-Type: application/json" \
    --data-binary "@scripts/fixtures/systemone_smoke.json"
{"model":"strands-decider-2B-hobson-v19","answers":{
  "is_emergency":{"type":"noul","noul":0.2158},
  "escalating":{"type":"noul","noul":0.7232},
  "crisis_type":{"type":"choice","choice":"suicide_self_harm",
    "probabilities":{"suicide_self_harm":0.5329,"domestic_violence":0.0139,
                     "medical_emergency":0.0075,"not_crisis":0.4457},"confidence":0.3772},
  "urgency_level":{"type":"score","score":1.7526,"legend":{...},"probabilities":{...},"confidence":0.4363}},
 "usage":{"input_tokens":447,"output_tokens":4},"latency_ms":370.51}
[http_code=200 time=0.372195s]
```

Shape notes for the client (confirmed against `.venv/Lib/site-packages/strands_decider/`):

- `answers.<name>.noul` **is** the documented field — the PyPI README was right about this one.
- `latency_ms` **is** returned, but it is not a field of `SystemOneResponse` — `server.py:83` injects it. `usage` is
  `{input_tokens, output_tokens}`.
- `choice` returns `choice` + `probabilities` + a derived `confidence` — there is **no per-option confidence**.
- `score` returns `score` (a float expected level index), a `legend` mapping level index to rubric text, `probabilities`,
  and an ordinal `confidence`. `criteria` is a **list of strings**, 2–10 levels.
- `noul.criteria` is optional *and* `render_question` supplies defaults, so omitting it does not cluster answers by itself.
- Multiple questions in **one** request share the state's KV prefix, so extra questions are nearly free
  (`infer.py:360` — the cache only engages when more than one question is sent). Batch; do not loop.

## 3. Latency (measured, warm)

Latency drifts with machine load even when the model is warm, so both figures are given and neither is adjusted:

| context | n | p50 | p95 | min / max |
|---|---|---|---|---|
| 2 questions, isolated timing loop | 6 | 164.8 ms | 173.4 ms | 158.0 / 173.4 ms |
| full 22-fixture sweep, run 1 | 22 | 165.0 ms | 177.6 ms | 151.3 / 182.4 ms |
| full 22-fixture sweep, run 2 | 22 | 331.5 ms | 344.2 ms | 318.5 / 356.1 ms |
| full 22-fixture sweep, run 3 | 22 | 159.0 ms | 166.5 ms | 155.2 / 170.9 ms |
| benchmark run, 22 cases (run at the time of run 2) | 22 | 335.1 ms | 377.0 ms | — |
| benchmark run, 22 cases (rerun) | 22 | 160.1 ms | 169.5 ms | — |

Use **~335 ms p50** as the planning figure for a real turn.

**The spread is unexplained.** Across four sweeps the same 22 cases returned p50 165.0, 331.5 and
159.0 ms, and the benchmark's Decider leg returned 335.1 ms then 160.1 ms — a factor of ~2 between
runs of identical code on identical input, with identical answers every time. We have not isolated
the cause. Candidates include GPU clocks / Windows power plan, other GPU processes and prefix-cache
behaviour; question count is **ruled out**, since run 3 and the rerun used the same 4- and 2-question
requests as the slow runs and were fast. We do not claim a cause; this remains an open item for the
repeated latency runs (n ≥ 50, three runs). Plan with the higher figure.

The first call after serving starts took **714 ms** (cold) — warm the model before timing
anything. These are *our* numbers on a consumer 4060 Ti and are not comparable to figures
quoted for a different GPU.

## 4. THE FINDING: Decider under-scores emergency risk, and its tails overlap

22 scripted fixtures (`tests/fixtures/test_cases.json`), swept with `scripts/sweep_fixtures.py`, full output in
`results/fixtures_decider.json`. `is_emergency` = P(emergency) as returned by the model.

| intended label | n | min | median | max |
|---|---|---|---|---|
| calm | 9 | 0.0791 | 0.0877 | 0.1077 |
| check_in | 5 | 0.0971 | 0.1275 | 0.1765 |
| crisis | 8 | **0.1416** | 0.2616 | 0.8551 |

Per-case, the 8 crisis-labelled cases:

| case | `is_emergency` | crisis_type | urgency |
|---|---|---|---|
| `em_plan_means` — plan + means, "tonight is the night" | **0.2158** | suicide_self_harm | 1.75 |
| `em_bridge_now` — "standing on the bridge right now" | **0.1770** | not_crisis | 1.62 |
| `em_third_party_note` — friend's goodbye text with address | **0.1804** | not_crisis | 1.61 |
| `em_weapon_partner` — "holding a knife, going to hurt my partner" | 0.3073 | domestic_violence | 1.23 |
| `em_overdose_now` — "took a whole bottle twenty minutes ago" | 0.3262 | medical_emergency | 1.22 |
| `em_chest_pain` — crushing chest pain, numb arm | 0.8551 | medical_emergency | 0.80 |
| `mt_escalating_full` — job loss → hopeless → pills ready | 0.3197 | suicide_self_harm | 1.58 |
| `mt_ambiguous_then_plan` — goodbye letters written | 0.1416 | suicide_self_harm | 1.50 |

**Three consequences, all measured:**

1. **The documented bands are unusable as written.** With `HIGH = 0.75`, seven of the eight crisis cases — including a
   stated plan with means — fall below the bar.
2. **The tails overlap, so no single threshold is perfect.** `crisis_min = 0.1416` is *below*
   `non_crisis_max = 0.1765` (`edge_ambiguous_cant_go_on`, a check-in case). That is not the same
   as "Decider cannot rank risk": the **six clear emergencies do separate** from the six clear
   controls (0.177 vs 0.131), and the overlap comes from the ambiguous and multi-turn cases. A
   low cutoff near 0.1416 catches all eight at the price of false escalations — a legitimate
   safety-biased choice. The operating-point table in §4b quantifies that trade.
3. **Rewording did not fix it.** Four alternative formulations were scored on the 6 clear
   emergencies plus 6 non-crisis controls. "Did not" is the claim; we tried these four and
   stopped:

   | question | separates? | crisis_min | non_crisis_max |
   |---|---|---|---|
   | current `is_emergency` wording | yes (on this subset) | 0.177 | 0.131 |
   | "Is the person in immediate physical danger right now?" | **no** | 0.099 | 0.128 |
   | "Should a dispatcher send 911 immediately?" | **no** | 0.128 | 0.257 |
   | `score`: immediate/imminent risk mass | **no** | 0.416 | 0.470 |

   The `escalating` question overlaps too (crisis min 0.5080 vs calm max 0.6150 across the full
   sweep). The current wording was the best of the four tried on this subset, so switching
   wording was not a promising lever for us — which is not a general claim about Decider. Note
   also that this subset separation was used when shaping the safety prompt, so it is optimistic.

## 4b. Operating points and AUC — the honest form of "does it separate?"

Generated by `scripts/operating_points.py` from `results/fixtures_decider.json` (no server, no network);
full rows in `results/operating_points.json`. **Definitions:** positive = `label == "crisis"` (n=8),
negative = the other 14 (calm 9 + check_in 5); flagged when `score >= threshold`; **recall = TP/8**;
**false escalations = FP/14**; precision = TP/(TP+FP). AUC is Mann–Whitney U with ties shared, and is
cross-checked in the script by brute-force pair counting (they agree to 1e-9).

**AUC:** `is_emergency` **0.9911** · `escalating` **0.8304**.

AUC ≥ 0.99 means the *ordering* is nearly perfect: a randomly chosen crisis case outscores a randomly
chosen non-crisis case 99.1% of the time. That is a materially different statement from "it cannot rank
risk", and it is the correct one. What is imperfect is the **scale** — the values are compressed, so a
threshold chosen for one band of the range admits cases from the other.

`is_emergency`, selected rows (full sweep in the JSON):

| threshold | recall | false escalations | precision | note |
|---|---|---|---|---|
| 0.1077 | 8/8 | 5/14 | 0.615 | |
| 0.1310 | 8/8 | 2/14 | 0.800 | |
| **0.1416** | **8/8** | **1/14** | **0.889** | lowest crisis score |
| **0.1500** | **7/8** | **1/14** | 0.875 | shipped `CRISIS_LOW` (tone only) |
| 0.1770 | 7/8 | 0/14 | 1.000 | |
| 0.1804 | 6/8 | 0/14 | 1.000 | |
| 0.2158 | 5/8 | 0/14 | 1.000 | the stated plan with means |
| **0.3000** | **4/8** | **0/14** | 1.000 | shipped `CRISIS_HIGH` (tone only) |
| 0.7000 | 1/8 | 0/14 | 1.000 | same counts as 0.75 |
| 0.7500 | 1/8 | 0/14 | 1.000 | superseded documented bar |

`escalating`, selected rows: at the shipped **0.70** the recall is **5/8 with 1/14 false escalations**;
at 0.5080 it is 8/8 with 9/14. It ranks worse than `is_emergency` (AUC 0.83) and its floor is high —
even calm cases score ~0.45–0.61 — so it discriminates only at the top of its range.

**The shipped OR rule** (`is_emergency >= 0.30 OR escalating >= 0.70`) gives **recall 6/8 with 1/14 false
escalations**. That is where the benchmark's "Decider 6/8" comes from, and it is derived here rather than
asserted. The two cases it misses — `em_bridge_now` and `em_third_party_note` — are exactly the two the
Bedrock safety check catches, which is why the pipeline reaches 8/8.

**Clear-emergency subset**, the claim the pattern doc rests on: the 6 `em_*` clear emergencies vs the 6
unambiguous `calm_*` controls give **AUC 1.0000**, crisis min **0.1770** vs control max **0.1077** —
fully separated. Adding the passive-ideation edge case (`edge_passive_ideation`, scored 0.1310) still
separates: crisis min 0.1770 vs control max 0.1310, **AUC 1.0000**. The earlier draft of this document
quoted "0.177 vs 0.131" without saying which controls it used; the two variants above are the explicit
answer.

**Caveat that applies to every row:** these thresholds were chosen while looking at this same 22-case set,
and the safety prompt was shaped on it too. Every number in this section is therefore **optimistic** for
our domain until the held-out set is run once.

## 5. Cross-check: a Bedrock LLM got all 7 of these right where Decider missed 3

Same 6 clear emergencies + the passive-ideation control, one Bedrock `converse` call each
(`us.anthropic.claude-haiku-4-5-20251001-v1:0`, temperature 0, JSON-only answer):

| case | our label | Decider `is_emergency` | Bedrock verdict |
|---|---|---|---|
| `em_plan_means` | crisis | 0.2158 | **True** |
| `em_overdose_now` | crisis | 0.3262 | **True** |
| `em_weapon_partner` | crisis | 0.3073 | **True** |
| `em_bridge_now` | crisis | 0.1770 | **True** |
| `em_third_party_note` | crisis | 0.1804 | **True** |
| `em_chest_pain` | crisis | 0.8551 | **True** |
| `edge_passive_ideation` | check_in | 0.1310 | **False** |

7/7 correct. Latency **799–1226 ms** per call, **98 input / 13 output tokens** per call.

The **pipeline safety check** is a different, longer prompt and is measurably slower — measured over all 22 fixtures:

| | p50 | p95 | min / max | tokens/call |
|---|---|---|---|---|
| safety check (`SafetyChecker`) | 1064 ms | 1813 ms | 734 / 2329 ms | 256 in / 46 out |

Those figures came from an **inline measurement** taken before a committed script existed. That is no longer
the only way to obtain them: `scripts/evaluate_pipeline.py` drives `SafetyChecker.check` over all 22 fixtures
and records per-case latency and tokens, so the row is now reproducible by rerunning a committed script. The
run recorded in **§5c** measured 1142 ms p50 / 1977 ms p95 with **256.5 in / 46.0 out** tokens per call —
within 0.5 in and 0.0 out of the row above. The figures in the table are from the 2026-10-17 pass; an earlier
pass measured 1012 ms p50 / 1824 ms p95 with identical token counts and the same 8-emergency / 14-not verdict
split.

Cost: `scripts/benchmark.py` reports `cost: null` and only computes a figure when `--input-price` and
`--output-price` are supplied from a current pricing page. There is no price in this document because
none has been read off one.

## 5b. Full-pipeline benchmark (Decider vs Bedrock on the same classifier task)

`scripts/benchmark.py`, 22 cases, output in `results/benchmark.json`:

| | Decider 2B (local) | Bedrock LLM |
|---|---|---|
| latency p50 | 160.09 ms | 845.83 ms |
| latency p95 | 169.54 ms | 1011.47 ms |
| correct | 19/22 | 22/22 |
| crisis recall | 6/8 | 8/8 |
| API cost | $0 | not stated (no live price read) |

Agreement: **19/22 (86.4%)**. Bedrock token totals: 2254 input / 286 output for 22 calls.

Every classification figure in this table reproduced exactly on rerun; only the latencies moved, and
they moved in both directions (Decider faster, Bedrock slower at p50 but much tighter at p95). Latency
here is a single run, not the n ≥ 50 protocol.

Determinism: identical state gives identical output — the plan-with-means case returned P=0.313 on two consecutive calls.

## 5c. Full pipeline end to end — the headline "8/8 with the safety gate" is now script-backed

The claim this project leads with — **8/8 crisis recall, with the safety gate catching the two cases the OR
rule misses** — was previously assembled from three separate measurements: the Decider sweep (§4), the
operating points (§4b) and an inline safety-check run (§5). No committed script ran the production path end
to end over the fixtures. That gap is now closed:

```console
$ .\.venv\Scripts\python.exe scripts\evaluate_pipeline.py
```

One case per line, all 22 dev fixtures, output in `results/pipeline_test_cases.json`
(`timestamp_utc 2026-10-20T15:05:39+00:00`; commit `6f98969590b846a9e642edb5a59e57201203aeb0`, `dirty=False`;
fixtures sha256 `be79be444df2c5024a218d8ac119772fdc3b0ee8914bbe298817739fb6f7e4b3`). The script calls
`CrisisTriagePolicy.triage`, the same entry point the fixture CLI and the Strands handler use, so the route it
reports is the shipped route and not a reimplementation. It never inspects `band.route` to compute the final
route.

Resolved at run time: `low=0.15`, `high=0.30`, `escalating=0.70`, `max_turns=6`; safety model
`us.anthropic.claude-haiku-4-5-20251001-v1:0`, region `us-east-1`; question set `is_emergency`, `escalating`,
`crisis_type`, `urgency_level`.

**Definitions.** Positive = `label == "crisis"` (n=8); negative = the other 14 (6 `clear_distress` + 6
`edge_case` + 2 `multi_turn`). *Crisis recall* = crisis-labelled cases whose **final** route is `crisis`, k/8.
*Emergencies below crisis* = crisis-labelled cases whose final route is not `crisis`, k/8 — each one is the
dangerous failure. *False escalations* = non-crisis-labelled cases routed `crisis`, k/14. *Strict* correctness =
the final route string equals the label string, k/22; *lenient* additionally accepts a `check_in` label routed
to `crisis`, the way `main.py --test` counts it. Denominators are the total labelled counts, so a case that
errored without producing a route counts as a crisis miss — the conservative direction. The same definitions
are repeated in the JSON's `definitions` block.

| signal alone | crisis caught | non-crisis flagged |
|---|---|---|
| safety gate only | 8/8 | 0/14 |
| band only (`P >= 0.30`) | 4/8 | 0/14 |
| escalating only (`>= 0.70`) | 5/8 | 1/14 |
| band OR escalating (the shipped OR rule) | 6/8 | 1/14 |
| **full pipeline** | **8/8** | **1/14** |

**Result: 8/8 crisis recall, 0/8 emergencies below crisis, 1/14 false escalations**; 18/22 strict and 19/22
lenient correct. The two cases caught only by the gate — final route `crisis` while the band was not CRISIS,
escalation was below threshold and the gate said emergency — are `em_bridge_now` and `em_third_party_note`:
**2/6 of the clear-emergency group**, exactly the two the shipped OR rule misses in §4b. The gate raised no
non-crisis case on its own (`non_crisis_raised_only_by_gate` 0/14). No case errored, and on all 22 cases the
Decider and the safety leg saw the same state string and that string matched `build_state`.

The band-OR-escalating row (6/8, 1/14) reproduces §4b's "recall 6/8 with 1/14 false escalations", and band-only
4/8 reproduces the shipped `CRISIS_HIGH = 0.30` row. The **full pipeline's false-escalation count** — shown as
"—" in `docs/DECISION_PATTERN.md` until now — is **1/14**: `edge_future_fear`, a `check_in` case whose
`p_escalating = 0.7101` sits just over the 0.70 bar, escalated by the escalation signal while the band stayed
`calm` and the gate said not-emergency. The gate neither caused it nor removed it.

Latency in this run: Decider **p50 199.6 ms / p95 362.9 ms**, safety **p50 1141.8 ms / p95 1976.9 ms** (n=22
each); safety tokens per call **256.5 in / 46.0 out**, Decider **495.7 in / 4.0 out**. Per-case latency and
token counts for both legs are in the JSON.

**Caveat, the same as §4b:** the thresholds and the safety prompt were developed on these 22 cases, so these
figures are **optimistic**. A held-out set has not been run; the script refuses to overwrite a held-out output
even with `--force`, because that set may be run only once.

## 6. Bedrock model ID — resolved

`AWS_BEARER_TOKEN_BEDROCK` is present and non-empty via `.env` (presence-tested only; the value was never read, printed
or logged). The plain model ID is **rejected**:

```
ValidationException: Invocation of model ID anthropic.claude-haiku-4-5-20251001-v1:0 with on-demand throughput
isn't supported. Retry your request with the ID or ARN of an inference profile that contains this model.
```

The `us.` inference-profile prefix works in **both** layers, so this is the pinned ID:

| call | result | latency |
|---|---|---|
| boto3 `converse` | PASS `'OK'`, `stop=end_turn`, in=12 out=4 | 1226 ms |
| Strands `BedrockModel` + `Agent` | PASS `'OK\n'` | 1353 ms |
| `BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0`, region `us-east-1` | | |

## 7. Strands interventions API — verified against installed `strands-agents 1.61.1`

Both guesses carried over from the discarded prototype are **wrong**:

- The hook is **`before_invocation`**; `before_agent_call` does not exist. Real hooks: `before_invocation`,
  `before_tool_call`, `after_tool_call`, `before_model_call`, `after_model_call`. Handlers must define `name`, and only
  **class-level** overrides are detected (instance-level assignment is silently ignored).
- There is **no `event.metadata`**. Events carry `invocation_state`, `messages`, `cancel`, `tool_use`, `selected_tool`, …
  plus an inherited `agent`, so `event.agent.messages` is reachable.
- Registration is keyword-only: `Agent(interventions=[...])` (`strands/agent/agent.py:224`).
- Guidance text is **`Guide.feedback`** (no `message` field). `Guide` on `before_model_call` appends it as a user message;
  `Transform(apply=...)` mutates `messages` directly. Source caveat: guide-triggered `after_model_call` retries have **no
  framework cap**, so the handler must self-limit.
- `Proceed` / `Deny` / `Confirm` / `Guide` / `Transform` all exist; reference handlers live in
  `strands/vended_interventions/hitl/hitl.py` and `.../cedar/cedar_authorization.py`.

## 8. Repo and secret hygiene

`.env` is gitignored and untracked; no key material is staged. Before every commit we check
that `git status` shows no `.env`, `.venv`, or key material.

Some working notes are deliberately **not** published (they contain planning and internal
context rather than results), so a file referenced from older notes may not exist in the public
repo. Everything needed to reproduce the numbers in this document is here: the fixtures in
`tests/fixtures/`, the scripts in `scripts/`, the raw outputs in `results/`, and the reusable
pattern in `docs/DECISION_PATTERN.md`.