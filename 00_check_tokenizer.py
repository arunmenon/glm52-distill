#!/usr/bin/env python3
"""
00_check_tokenizer.py — HARD GATE for logit KD.

Compares teacher and student vocabularies and decides the KD mode:

  exact  — identical token->id mapping (or student is a clean superset):
           teacher token ids can be reused directly.
  remap  — vocabs share (nearly) all tokens by STRING but some ids differ
           (e.g. GLM-5.2 vs GLM-4.5-Air: 36 specials incl. <think> moved,
           plus teacher-only filler/audio tokens). 03_pack_dataset.py builds
           a teacher-id -> student-id table, drops samples whose responses
           contain unmappable tokens, and masks unmappable top-K slots.
  none   — vocabs differ too much; Student A falls back to SFT-only.

Reads vocab straight from tokenizer.json (works regardless of the
transformers version / custom tokenizer classes); falls back to
AutoTokenizer when a model ships no tokenizer.json.

Writes gate_result.json (consumed by run_all.sh and 03_pack_dataset.py).
"""

import argparse
import json
from pathlib import Path

# unmappable teacher-vocab fraction above which remap mode is refused
REMAP_MAX_MISSING_FRAC = 0.10


def load_vocab(model_dir: str) -> dict:
    p = Path(model_dir) / "tokenizer.json"
    if p.exists():
        d = json.loads(p.read_text())
        vocab = dict(d["model"]["vocab"])
        for t in d.get("added_tokens", []):
            vocab[t["content"]] = t["id"]
        return vocab
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True).get_vocab()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student-a", required=True)
    ap.add_argument("--out", default="gate_result.json")
    args = ap.parse_args()

    tv, sv = load_vocab(args.teacher), load_vocab(args.student_a)

    only_t = set(tv) - set(sv)
    only_s = set(sv) - set(tv)
    shared = set(tv) & set(sv)
    id_mismatch = [k for k in shared if tv[k] != sv[k]]
    missing_frac = len(only_t) / max(len(tv), 1)

    if not only_t and not id_mismatch:
        mode = "exact"          # ids reusable directly (superset students included)
    elif missing_frac <= REMAP_MAX_MISSING_FRAC:
        mode = "remap"          # string-keyed id remap covers (nearly) everything
    else:
        mode = "none"

    detail = {
        "teacher_vocab": len(tv),
        "student_vocab": len(sv),
        "tokens_only_in_teacher": len(only_t),
        "tokens_only_in_student": len(only_s),
        "shared_tokens_with_different_ids": len(id_mismatch),
        "unmappable_teacher_vocab_frac": round(missing_frac, 4),
        "id_mismatch_examples": sorted(id_mismatch)[:10],
    }
    if mode == "remap":
        detail["note"] = (
            "ids remapped at pack time; 03_pack_dataset.py prints the sample "
            "drop rate from unmappable response tokens — verify it stays <1%"
        )

    result = {"logit_kd_ok": mode != "none", "kd_mode": mode, "detail": detail}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"LOGIT_KD_OK={1 if result['logit_kd_ok'] else 0} KD_MODE={mode}")


if __name__ == "__main__":
    main()
