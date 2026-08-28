#!/usr/bin/env python3
"""Docker Hub image preflight over the expansion candidate pool.

One timestamped, auditable snapshot for ALL candidates (seeds included):
per task the exact image reference, HTTP result, failure reason, retry
history, and a three-way classification:
  available     — tags endpoint returned 200
  missing       — 404 on every attempt (image genuinely absent)
  indeterminate — network/rate-limit/5xx after retries; NOT proof of absence

No candidates are replaced here; output is the live-candidate inventory
plus repo-level shortfall so refill can be sized afterwards.

Input : runs/expansion/bugfix/candidate_pool.json
Output: runs/expansion/bugfix/image_preflight.json
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).parent
POOL_FILE = REPO_DIR / "runs" / "expansion" / "bugfix" / "candidate_pool.json"
OUT_FILE = REPO_DIR / "runs" / "expansion" / "bugfix" / "image_preflight.json"
MAX_ATTEMPTS = 3
RETRY_SLEEP_S = 5.0
TIMEOUT_S = 20


def image_name(instance_id: str) -> str:
    return f"xingyaoww/sweb.eval.x86_64.{instance_id.replace('__', '_s_')}".lower()


def probe(image: str) -> dict:
    url = f"https://hub.docker.com/v2/repositories/{image}/tags?page_size=1"
    attempts = []
    for n in range(1, MAX_ATTEMPTS + 1):
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
                attempts.append({"n": n, "at": stamp,
                                 "http_status": resp.status})
                if resp.status == 200:
                    return {"result": "available", "attempts": attempts}
        except urllib.error.HTTPError as e:
            attempts.append({"n": n, "at": stamp, "http_status": e.code,
                             "reason": str(e.reason)})
            if e.code == 404:
                return {"result": "missing", "attempts": attempts}
            # 429/5xx: retry — transient, not evidence of absence
        except Exception as e:
            attempts.append({"n": n, "at": stamp, "http_status": None,
                             "reason": f"{type(e).__name__}: {e}"})
        if n < MAX_ATTEMPTS:
            time.sleep(RETRY_SLEEP_S)
    return {"result": "indeterminate", "attempts": attempts}


def main():
    pool = json.loads(POOL_FILE.read_text())
    records, counts = [], {"available": 0, "missing": 0, "indeterminate": 0}
    for i, task in enumerate(pool, 1):
        iid = task["instance_id"]
        image = image_name(iid)
        r = probe(image)
        counts[r["result"]] += 1
        records.append({
            "instance_id": iid, "repo": task["repo"], "image": image,
            "candidate_source": task.get("candidate_source"),
            "review_status": task.get("review_status"),
            "heuristic_difficulty": task.get("difficulty"),
            "result": r["result"], "attempts": r["attempts"],
        })
        print(f"[{i:2d}/{len(pool)}] {r['result']:13s} {iid}")

    by_repo = {}
    for rec in records:
        d = by_repo.setdefault(rec["repo"], {"available": 0, "missing": 0,
                                             "indeterminate": 0})
        d[rec["result"]] += 1

    report = {
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "pool_file": str(POOL_FILE.relative_to(REPO_DIR)),
        "pool_sha256_expected":
            "4daf5a7dceed50161e23e0eb8de19b8abc45f27e7e95e58d81c24cad41ef106d",
        "registry": "hub.docker.com xingyaoww/sweb.eval.x86_64.*",
        "policy": {"max_attempts": MAX_ATTEMPTS,
                   "retry_sleep_s": RETRY_SLEEP_S,
                   "classification": {
                       "available": "HTTP 200 on tags endpoint",
                       "missing": "HTTP 404 (terminal on first sight)",
                       "indeterminate":
                           "non-404 failure on all attempts; re-probe "
                           "before treating as absent"}},
        "totals": counts,
        "repo_shortfall": by_repo,
        "records": records,
    }
    OUT_FILE.write_text(json.dumps(report, indent=1))
    print(json.dumps({"totals": counts, "repo_shortfall": by_repo}, indent=1))
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
