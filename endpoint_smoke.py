#!/usr/bin/env python3
"""Public-endpoint smoke test for the agentic trajectory leg.

Answers the open questions in agentic_trajectory_design.md section 1a, per
candidate provider serving z-ai/glm-5.2 on OpenRouter:

  1. PINNING     provider.order + allow_fallbacks=false actually pins?
  2. REASONING   thinking returned, and via which field?
  3. LOGPROBS    top-20 returned? do they cover reasoning tokens or only the
                 answer channel? (decides logit-KD via API vs rescore pass)
  4. TOKEN IDS   do logprob token strings round-trip to single GLM-5.2 ids?
  5. CACHING     does a repeated long prefix earn a cache-read discount?

Usage:
  OPENROUTER_API_KEY=... [GLM_TOKENIZER_JSON=path] python3 endpoint_smoke.py
  (key falls back to .env.openrouter next to this script)

Writes endpoint_smoke_report.json (gitignored). Budget: well under $1; the
script prints accumulated cost as it goes and aborts past $2.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "z-ai/glm-5.2"
# fp8-or-better + top_logprobs=true per the 2026-07-22 endpoint survey
PROVIDERS = ["StreamLake", "GMICloud", "Fireworks", "Alibaba"]
COST_ABORT_USD = 2.0

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "endpoint_smoke_report.json")

spent_usd = 0.0


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".env.openrouter")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("no OPENROUTER_API_KEY in env or .env.openrouter")
    return key


API_KEY = load_api_key()


def load_tokenizer():
    path = os.environ.get("GLM_TOKENIZER_JSON", "")
    if not path or not os.path.exists(path):
        return None
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(path)
    except Exception as exc:  # noqa: BLE001 - report-only tool
        print(f"tokenizer unavailable ({exc}); TOKEN IDS test will be SKIPPED")
        return None


def chat(payload: dict, timeout: int = 180) -> dict:
    global spent_usd
    if spent_usd > COST_ABORT_USD:
        sys.exit(f"cost abort: ${spent_usd:.3f} > ${COST_ABORT_USD}")
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return {"_http_error": err.code, "_body": err.read().decode()[:500]}
    except Exception as exc:  # noqa: BLE001 - record and continue
        return {"_error": str(exc)[:300]}
    cost = (data.get("usage") or {}).get("cost")
    if isinstance(cost, (int, float)):
        spent_usd += cost
    return data


def pinned(provider: str) -> dict:
    return {"order": [provider], "allow_fallbacks": False}


def base_payload(provider: str) -> dict:
    return {"model": MODEL, "provider": pinned(provider),
            "usage": {"include": True}}


def logprob_probe(provider: str) -> dict:
    """Tests 1-4: one thinking-triggering completion with top-20 logprobs."""
    result = {}
    payload = base_payload(provider) | {
        "messages": [{"role": "user",
                      "content": "What is 17*23? Think step by step, then "
                                 "give the final number."}],
        "max_tokens": 1500, "temperature": 0.7,
        "logprobs": True, "top_logprobs": 20,
        "reasoning": {"enabled": True},
    }
    data = chat(payload)
    if "_http_error" in data or "_error" in data:
        # some providers reject the reasoning param or top_logprobs=20;
        # distinguish by retrying reduced variants
        retry = chat({k: v for k, v in payload.items() if k != "reasoning"})
        if "_http_error" in retry or "_error" in retry:
            retry5 = chat({**payload, "top_logprobs": 5})
            if "_http_error" in retry5 or "_error" in retry5:
                result["error"] = data
                return result
            result["note"] = "top_logprobs=20 rejected, 5 accepted"
            data = retry5
        else:
            result["note"] = "reasoning param rejected, retried without"
            data = retry

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}

    result["served_by"] = data.get("provider")
    result["pinned_ok"] = (data.get("provider") == provider)
    result["finish_reason"] = choice.get("finish_reason")

    reasoning_text = (message.get("reasoning")
                      or message.get("reasoning_content") or "")
    content_text = message.get("content") or ""
    result["reasoning_field"] = (
        "reasoning" if message.get("reasoning") else
        "reasoning_content" if message.get("reasoning_content") else
        "inline_think" if "<think>" in content_text else "ABSENT")
    result["reasoning_chars"] = len(reasoning_text)
    result["content_chars"] = len(content_text)

    lp_entries = (choice.get("logprobs") or {}).get("content") or []
    result["n_logprob_positions"] = len(lp_entries)
    if lp_entries:
        top_counts = [len(e.get("top_logprobs") or []) for e in lp_entries]
        result["top_logprobs_per_position_min_max"] = (min(top_counts),
                                                       max(top_counts))
        result["bytes_field_present"] = all(
            e.get("bytes") is not None for e in lp_entries[:50])

    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = details.get("reasoning_tokens")
    result["completion_tokens"] = completion_tokens
    result["reasoning_tokens"] = reasoning_tokens
    n_lp = len(lp_entries)
    if n_lp and completion_tokens:
        if n_lp >= 0.95 * completion_tokens:
            result["logprob_coverage"] = "FULL (reasoning + answer)"
        elif (reasoning_tokens
              and abs(n_lp - (completion_tokens - reasoning_tokens))
              <= 0.05 * completion_tokens):
            result["logprob_coverage"] = "CONTENT-ONLY (reasoning uncovered)"
        else:
            result["logprob_coverage"] = f"UNCLEAR (n={n_lp} vs completion=" \
                                         f"{completion_tokens})"
    else:
        result["logprob_coverage"] = "NO LOGPROBS" if not n_lp else "UNKNOWN"

    if TOKENIZER and lp_entries:
        sample = lp_entries[:200]
        single = 0
        for entry in sample:
            token_text = entry.get("token", "")
            if entry.get("bytes"):
                try:
                    token_text = bytes(entry["bytes"]).decode(
                        "utf-8", errors="surrogateescape")
                except Exception:  # noqa: BLE001
                    pass
            ids = TOKENIZER.encode(token_text,
                                   add_special_tokens=False).ids
            if len(ids) == 1:
                single += 1
        result["token_id_single_roundtrip_pct"] = round(
            100.0 * single / len(sample), 1)
    else:
        result["token_id_single_roundtrip_pct"] = "SKIPPED"

    result["cost_usd"] = usage.get("cost")
    return result


CACHE_FILLER = ("You are a software engineering agent operating in a bash "
                "environment inside a task container. Follow the repository "
                "conventions, run tests after every change, and keep edits "
                "minimal and reviewable. ") * 180  # ~3-4k tokens of prefix


def cache_probe(provider: str) -> dict:
    """Test 5: identical long-prefix call twice; look for cache-read credit."""
    result = {}
    payload = base_payload(provider) | {
        "messages": [{"role": "system", "content": CACHE_FILLER},
                     {"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 400, "temperature": 0.0,
    }
    runs = []
    for attempt in ("cold", "warm"):
        data = chat(payload)
        if "_http_error" in data or "_error" in data:
            result[f"{attempt}_error"] = data
            return result
        usage = data.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        runs.append({"attempt": attempt,
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "cached_tokens": prompt_details.get("cached_tokens"),
                     "cost_usd": usage.get("cost")})
        time.sleep(2)
    result["runs"] = runs
    warm_cached = runs[-1].get("cached_tokens") or 0
    cold_cost, warm_cost = runs[0].get("cost_usd"), runs[-1].get("cost_usd")
    result["cache_discount"] = (
        "YES" if warm_cached > 0 else
        "COST-ONLY" if (cold_cost and warm_cost and warm_cost < 0.7 * cold_cost)
        else "NOT OBSERVED")
    return result


TOKENIZER = load_tokenizer()


def main() -> None:
    report = {"model": MODEL, "date": time.strftime("%Y-%m-%d %H:%M"),
              "providers": {}}
    for provider in PROVIDERS:
        print(f"\n=== {provider} ===")
        entry = {"logprob_probe": logprob_probe(provider)}
        print(json.dumps(entry["logprob_probe"], indent=2)[:1200])
        entry["cache_probe"] = cache_probe(provider)
        print(json.dumps(entry["cache_probe"], indent=2)[:600])
        report["providers"][provider] = entry
        print(f"[spent so far: ${spent_usd:.4f}]")
    report["total_cost_usd"] = round(spent_usd, 4)
    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport -> {REPORT_PATH}  total ${spent_usd:.4f}")


if __name__ == "__main__":
    main()
