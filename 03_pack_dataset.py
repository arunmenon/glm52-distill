#!/usr/bin/env python3
"""
03_pack_dataset.py — Turn the generated corpus into training-ready HF datasets.

Outputs:
  packed/student_a/   (GLM-family, tokenizer-matched)  columns:
      input_ids, labels, topk_ids [L,K], topk_logprobs [L,K]
      -> input_ids = chat_template_prefix_ids + TEACHER's raw response_token_ids
         (token-exact: zero re-tokenization drift; topk arrays align 1:1 with labels)
  packed/student_b/   (Qwen3, mismatched tokenizer)     columns:
      input_ids, labels        (sequence-level KD only; response re-tokenized)

Alignment contract consumed by 04_train_kd.py:
  topk_ids[i] / topk_logprobs[i] describe the teacher distribution over labels[i].
  Prompt positions carry labels=-100, topk_ids=-1, topk_logprobs=-1e9.

KD modes (from gate_result.json, produced by 00_check_tokenizer.py):
  exact — teacher ids reused as-is.
  remap — teacher ids translated via a string-keyed teacher->student table
          (built from both tokenizer.json files). Samples whose responses
          contain unmappable tokens are dropped (rate printed — verify <1%);
          unmappable top-K slots are masked to -1 / -1e9.
  none  — student_a packed via the text path (SFT only, no top-K columns).

Also hard-fails if any held-out prompt appears in the training corpus.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

SEED = 42
K = 20
IGNORE = -100


def norm_hash(text: str) -> str:
    return hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()


def load_corpus(corpus_dirs):
    """Load and merge one or more corpus dirs (base + extensions).
    sample_ids are disjoint by construction (01b assigns offset ids)."""
    rows = []
    for corpus_dir in corpus_dirs:
        n0 = len(rows)
        for shard in sorted(Path(corpus_dir).glob("shard_*.parquet")):
            rows.extend(pq.read_table(shard).to_pylist())
        print(f"loaded {len(rows) - n0} corpus rows from {corpus_dir}")
    ids = [r["sample_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate sample_ids across corpus dirs!"
    return rows


def _flat_ids(out):
    """apply_chat_template(tokenize=True) returns a raw id list on
    transformers 4.x but a BatchEncoding on 5.x — normalize to a flat list."""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def prefix_ids_for(tok, prompt: str):
    return _flat_ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    ))


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


def build_remap(teacher_dir: str, student_dir: str) -> dict:
    """teacher token id -> student token id, keyed by token string."""
    tv, sv = load_vocab(teacher_dir), load_vocab(student_dir)
    remap = {tid: sv[tok_str] for tok_str, tid in tv.items() if tok_str in sv}
    n_identity = sum(1 for k, v in remap.items() if k == v)
    print(f"remap table: {len(remap)}/{len(tv)} teacher ids mappable "
          f"({n_identity} identical, {len(remap) - n_identity} moved, "
          f"{len(tv) - len(remap)} unmappable)")
    return remap


def build_student_a(rows, tok, max_len, remap=None):
    """Token-exact path: reuse teacher response token ids (remapped if needed)."""
    eos = tok.eos_token_id
    out = []
    skipped = 0
    dropped_unmappable = 0
    for r in rows:
        prefix = prefix_ids_for(tok, r["prompt"])
        resp = list(r["response_token_ids"])
        tids = [list(x) for x in r["topk_ids"]]
        tlps = [list(x) for x in r["topk_logprobs"]]
        if len(resp) != len(tids):  # defensive; should never happen
            skipped += 1
            continue
        if remap is not None:
            if any(t not in remap for t in resp):
                dropped_unmappable += 1
                continue
            resp = [remap[t] for t in resp]
            for ids_k, lps_k in zip(tids, tlps):
                for j, t in enumerate(ids_k):
                    if t in remap:
                        ids_k[j] = remap[t]
                    else:               # -1 pads and unmappable teacher tokens
                        ids_k[j] = -1
                        lps_k[j] = -1e9
        if eos is not None and (not resp or resp[-1] != eos):
            resp.append(eos)
            tids.append([-1] * K)      # no teacher dist for appended eos
            tlps.append([-1e9] * K)

        input_ids = prefix + resp
        labels = [IGNORE] * len(prefix) + resp
        topk_ids = [[-1] * K] * len(prefix) + tids
        topk_lps = [[-1e9] * K] * len(prefix) + tlps

        if len(input_ids) > max_len:   # drop overlong rather than truncate mid-thought
            skipped += 1
            continue
        out.append(
            {"input_ids": input_ids, "labels": labels,
             "topk_ids": topk_ids, "topk_logprobs": topk_lps}
        )
    n_in = max(len(rows), 1)
    print(f"student_a: kept {len(out)}, skipped {skipped}, "
          f"dropped_unmappable {dropped_unmappable} "
          f"({dropped_unmappable / n_in:.2%} — verify this stays <1%)")
    return out


def student_think_tags(tok):
    """Return (open, close) reasoning delimiters in the STUDENT's own vocab, or
    None if the student has no think convention. Detected from special tokens /
    added-tokens so we distill into the format the student natively emits,
    rather than flattening the teacher's reasoning into plain answer text
    (the cause of the v0 regression, journal Discovery #6)."""
    vocab = tok.get_vocab()
    for open_t, close_t in (("<think>", "</think>"),
                            ("<reasoning>", "</reasoning>")):
        if open_t in vocab and close_t in vocab:
            return open_t, close_t
    return None


def build_student_b(rows, tok, max_len):
    """Sequence-level path: re-tokenize teacher text with the student tokenizer,
    preserving reasoning inside the student's native think delimiters."""
    tags = student_think_tags(tok)
    verified_tags = False   # confirm the template keeps our think block once
    out, skipped, n_thinky = [], 0, 0
    for r in rows:
        reasoning = (r.get("reasoning_text") or "").strip()
        answer = (r["response_text"] or "").strip()
        if reasoning and tags:
            # Emit in the student's own format: <think>…</think>\n\n<answer>.
            open_t, close_t = tags
            text = f"{open_t}\n{reasoning}\n{close_t}\n\n{answer}"
            n_thinky += 1
        elif reasoning:
            # Student has no think convention: keep reasoning but plainly
            # (documented degradation, not a silent flatten).
            text = f"{reasoning}\n\n{answer}"
        else:
            text = answer
        msgs = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": text}]
        full = _flat_ids(tok.apply_chat_template(msgs, tokenize=True))
        prefix = prefix_ids_for(tok, r["prompt"])
        if full[: len(prefix)] != prefix:
            raise ValueError(
                "chat-template prefix is not a prefix of the full conversation "
                "tokenization — labels would silently misalign. Inspect this "
                f"tokenizer's chat template (sample_id={r['sample_id']})."
            )
        if len(full) <= len(prefix) or len(full) > max_len:
            skipped += 1
            continue
        # One-time guard: some chat templates strip <think> from assistant
        # history. If ours does, our reasoning is silently deleted — fail loud.
        if reasoning and tags and not verified_tags:
            close_id = tok.convert_tokens_to_ids(tags[1])
            if close_id not in full[len(prefix):]:
                raise ValueError(
                    f"student chat template dropped the {tags[1]} tag from the "
                    "assistant turn — reasoning would be silently lost. This "
                    "template needs enable_thinking or a manual template edit.")
            verified_tags = True
        labels = [IGNORE] * len(prefix) + full[len(prefix):]
        out.append({"input_ids": full, "labels": labels})
    tag_note = f"native think tags {tags}" if tags else "NO think tags (plain)"
    print(f"student_b: kept {len(out)}, skipped {skipped}, "
          f"{n_thinky} with reasoning wrapped in {tag_note}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+",
                    help="one or more corpus dirs (base + extensions from 01b)")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--teacher", default=None,
                    help="teacher model dir (tokenizer files suffice); required "
                         "when the gate decided kd_mode=remap")
    ap.add_argument("--student-a", required=True)
    ap.add_argument("--student-b", required=True)
    ap.add_argument("--out", default="packed/")
    ap.add_argument("--exclude-report", default=None,
                    help="benchmark_audit.py report; contaminated sample_ids "
                         "are dropped before packing")
    ap.add_argument("--max-len", type=int, default=8192,
                    help="drop samples longer than this many tokens; lower to "
                         "6144 if training OOMs")
    ap.add_argument("--gate", default="gate/gate_result.json",
                    help="gate_result.json from 00_check_tokenizer.py; if missing "
                         "or logit_kd_ok=false, student_a is packed via the text "
                         "path (no top-K columns) instead of token-exact reuse")
    args = ap.parse_args()

    kd_mode = "none"
    try:
        gate = json.load(open(args.gate))
        # older gate files have only logit_kd_ok; treat ok as exact
        kd_mode = gate.get("kd_mode", "exact" if gate["logit_kd_ok"] else "none")
    except Exception:
        print(f"WARNING: could not read {args.gate}; "
              "packing student_a WITHOUT logit-KD columns (safe default)")
    print(f"kd_mode={kd_mode} -> student_a path: "
          f"{'token-exact + top-K' if kd_mode != 'none' else 'text re-tokenization (SFT only)'}")

    remap = None
    if kd_mode == "remap":
        assert args.teacher, "kd_mode=remap requires --teacher <teacher model dir>"
        remap = build_remap(args.teacher, args.student_a)

    rows = load_corpus(args.corpus)

    if args.exclude_report:
        rep = json.load(open(args.exclude_report))
        bad = {c["sample_id"] for c in rep["contaminated"]}
        n0 = len(rows)
        rows = [r for r in rows if r["sample_id"] not in bad]
        print(f"excluded {n0 - len(rows)} benchmark-contaminated rows "
              f"({len(bad)} flagged in report)")

    # ---- contamination gate ----
    held_hashes = {
        norm_hash(json.loads(l)["prompt"]) for l in open(args.heldout)
    }
    n_bad = sum(1 for r in rows if norm_hash(r["prompt"]) in held_hashes)
    assert n_bad == 0, f"CONTAMINATION: {n_bad} held-out prompts found in corpus!"
    print("contamination gate passed (0 overlaps)")

    out = Path(args.out)
    if kd_mode != "none":
        builder_a = lambda rows, tok, ml: build_student_a(rows, tok, ml, remap=remap)  # noqa: E731
    else:
        builder_a = build_student_b
    for name, model_path, builder in [
        ("student_a", args.student_a, builder_a),
        ("student_b", args.student_b, build_student_b),
    ]:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        examples = builder(rows, tok, args.max_len)
        ds = Dataset.from_list(examples).shuffle(seed=SEED)
        split = ds.train_test_split(test_size=0.02, seed=SEED)
        dd = DatasetDict({"train": split["train"], "validation": split["test"]})
        dest = out / name
        dd.save_to_disk(str(dest))
        print(f"saved {dest}: train={len(dd['train'])}, val={len(dd['validation'])}")


if __name__ == "__main__":
    main()
