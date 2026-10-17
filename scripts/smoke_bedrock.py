"""Bedrock smoke test: settle the model ID before building on it.

Two layers, because they can disagree:
  1. boto3 `converse`      -- proves the key, region and model ID work at the API level.
  2. Strands `BedrockModel` -- proves the SDK wrapper we will actually use accepts it.

`.env` is loaded before any client is constructed: boto3 does not read `.env` itself.
The key is never printed or logged -- only its presence is reported.

    .\\.venv\\Scripts\\python.exe scripts/smoke_bedrock.py
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Plain ID first, then the inference-profile prefixes, then the pinned fallback.
CANDIDATES = [
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
]

PROMPT = "Reply with exactly: OK"


def report_env() -> None:
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    print(f"AWS_BEARER_TOKEN_BEDROCK set: {key is not None} (length>0: {bool(key)})")
    print(f"region: {REGION}")
    if not key:
        print("FAIL: no Bedrock key in the environment or .env -- cannot continue.")
        raise SystemExit(2)


def try_boto3(model_id: str) -> tuple[bool, str, float]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 0}),
    )
    started = time.perf_counter()
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 16, "temperature": 0},
        )
    except Exception as exc:  # noqa: BLE001 - the error text is the result
        return False, f"{type(exc).__name__}: {str(exc)[:300]}", (time.perf_counter() - started) * 1000
    elapsed = (time.perf_counter() - started) * 1000
    text = "".join(
        b.get("text", "") for b in resp["output"]["message"]["content"] if isinstance(b, dict)
    ).strip()
    usage = resp.get("usage", {})
    return (
        True,
        f"-> {text!r}  stop={resp.get('stopReason')}  "
        f"in={usage.get('inputTokens')} out={usage.get('outputTokens')}",
        elapsed,
    )


def try_strands(model_id: str) -> tuple[bool, str, float]:
    from strands import Agent
    from strands.models import BedrockModel

    started = time.perf_counter()
    try:
        model = BedrockModel(model_id=model_id, region_name=REGION, temperature=0.0)
        agent = Agent(model=model, system_prompt="You are terse. Follow instructions exactly.")
        result = agent(PROMPT)
    except Exception as exc:  # noqa: BLE001 - the error text is the result
        return False, f"{type(exc).__name__}: {str(exc)[:300]}", (time.perf_counter() - started) * 1000
    elapsed = (time.perf_counter() - started) * 1000
    return True, f"-> {str(result)[:200]!r}", elapsed


def main() -> int:
    report_env()
    print()
    winner: str | None = None

    for model_id in CANDIDATES:
        print(f"=== {model_id}")
        ok, detail, ms = try_boto3(model_id)
        print(f"  boto3 converse : {'PASS' if ok else 'FAIL'}  {ms:.0f} ms  {detail}")
        if not ok:
            continue
        s_ok, s_detail, s_ms = try_strands(model_id)
        print(f"  Strands BedrockModel: {'PASS' if s_ok else 'FAIL'}  {s_ms:.0f} ms  {s_detail}")
        if s_ok:
            winner = model_id
            print(f"\nRESOLVED: both layers work with {model_id}")
            break
        print("  (boto3 works but the Strands wrapper does not -- trying next candidate)")
        print()

    if winner is None:
        print("\nRESOLVED: no candidate passed both layers. Do not build on Bedrock yet.")
        return 1
    print(f'Set BEDROCK_MODEL_ID={winner} and AWS_DEFAULT_REGION={REGION}')
    return 0


if __name__ == "__main__":
    sys.exit(main())