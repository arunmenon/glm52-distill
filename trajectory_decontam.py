#!/usr/bin/env python3
"""Trajectory decontamination gate (agentic_trajectory_design.md section 3,
trajectory_task_spec.md section 7): assert the trajectory task list is
disjoint from SWE-bench Verified at THREE levels before any teacher spend:

  1. instance-level : no shared instance_id
  2. repo-level     : no shared repository (training-norm contamination;
                      sources claim this by construction, we check anyway)
  3. text-level     : no near-identical problem statements
                      (normalized exact hash + 8-gram containment)

Usage: python3 trajectory_decontam.py <task_list.json>
Writes trajectory_gate.json; exits non-zero on any hit (hard gate).
"""

import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).parent
GATE_FILE = REPO_DIR / "trajectory_gate.json"
VERIFIED_PARQUET = ("https://huggingface.co/datasets/princeton-nlp/"
                    "SWE-bench_Verified/resolve/main/data/test-00000-of-"
                    "00001.parquet")
CACHE = Path("/private/tmp/claude-501/-Users-arunmenon-projects-glm52-distill/"
             "ad896283-5c39-4a76-bdab-80daf2f812f4/scratchpad/"
             "swebench_verified.parquet")
NGRAM = 8
CONTAINMENT_THRESHOLD = 0.6


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def ngrams(text: str) -> set:
    words = norm(text).split()
    return {" ".join(words[i:i + NGRAM])
            for i in range(len(words) - NGRAM + 1)}


def main():
    import pandas as pd
    task_list = json.loads(Path(sys.argv[1]).read_text())
    if not CACHE.exists():
        urllib.request.urlretrieve(VERIFIED_PARQUET, CACHE)
    verified = pd.read_parquet(CACHE)

    v_ids = set(verified.instance_id)
    v_repos = set(verified.repo)
    v_hashes = {hashlib.md5(norm(s).encode()).hexdigest()
                for s in verified.problem_statement}
    v_ngrams = [(iid, ngrams(s)) for iid, s in
                zip(verified.instance_id, verified.problem_statement)]

    hits = {"instance": [], "repo": [], "text_exact": [], "text_ngram": []}
    for task in task_list:
        iid, repo = task["instance_id"], task["repo"]
        statement = task.get("problem_statement", "")
        if iid in v_ids:
            hits["instance"].append(iid)
        if repo in v_repos:
            hits["repo"].append(f"{iid} (repo {repo})")
        if hashlib.md5(norm(statement).encode()).hexdigest() in v_hashes:
            hits["text_exact"].append(iid)
        else:
            grams = ngrams(statement)
            if grams:
                for v_iid, vg in v_ngrams:
                    containment = len(grams & vg) / len(grams)
                    if containment >= CONTAINMENT_THRESHOLD:
                        hits["text_ngram"].append(
                            f"{iid} ~ {v_iid} ({containment:.2f})")
                        break

    clean = not any(hits.values())
    gate = {"date": time.strftime("%Y-%m-%d"),
            "task_list": sys.argv[1], "n_tasks": len(task_list),
            "against": "princeton-nlp/SWE-bench_Verified "
                       f"({len(verified)} instances)",
            "checks": {"instance_ids": len(hits["instance"]),
                       "repo_overlap": len(hits["repo"]),
                       "text_exact": len(hits["text_exact"]),
                       "text_ngram": len(hits["text_ngram"])},
            "hits": {k: v for k, v in hits.items() if v},
            "verdict": "CLEAN" if clean else "CONTAMINATED"}
    GATE_FILE.write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
