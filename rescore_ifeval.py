#!/usr/bin/env python3
"""
rescore_ifeval.py — re-score lm-eval IFEval samples after stripping the think
block (journal Discovery #6).

Why: IFEval checks the RAW response against instructions like "no commas",
"exactly N paragraphs", "wrap in double quotes". A thinking model emits a long
<think>…</think> block before the answer, which fails those checks no matter
how good the final answer is — so lm-eval's raw IFEval score for a reasoning
model is an artifact. This strips the think block from each logged response and
re-applies IFEval's OWN instruction classes (vendored in lm_eval), reporting
strict + loose prompt/instruction accuracy on the cleaned text.

Runs on-pod (needs lm_eval installed), CPU, seconds. Degrades loudly if the
lm-eval IFEval internals differ from what it expects — it never silently
reports a wrong number.

  python rescore_ifeval.py --samples <OUTDIR>/<model>/**/samples_ifeval_*.jsonl
"""

import argparse
import glob
import json
import re


def strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def load_instruction_registry():
    """Import lm-eval's vendored IFEval instruction classes. Raises with a
    clear message if the internal path changed (version drift)."""
    try:
        from lm_eval.tasks.ifeval import instructions_registry as reg
        return reg.INSTRUCTION_DICT
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not import lm_eval IFEval instructions ({e}); "
            "this lm-eval version organizes IFEval differently — inspect "
            "lm_eval/tasks/ifeval/ and update this import.")


def get_resp(sample):
    for key in ("filtered_resps", "resps"):
        v = sample.get(key)
        if not v:
            continue
        x = v[0]
        while isinstance(x, list):
            x = x[0]
        if isinstance(x, str):
            return x
    # some versions nest under "arguments"/"target"; fall back to a scan
    raise KeyError(f"no response field in sample keys={list(sample)}")


def get_doc(sample):
    return sample.get("doc", sample)


def check_one(inst_dict, doc, response):
    """Return (strict_ok, loose_ok) for one prompt across its instructions."""
    ids = doc["instruction_id_list"]
    kwargs_list = doc.get("kwargs") or [{}] * len(ids)
    prompt = doc.get("prompt", "")
    strict, loose = [], []
    # loose variants: IFEval tries several response transforms and passes if ANY
    loose_variants = [
        response,
        response.replace("*", ""),
        "\n".join(line for line in response.split("\n") if line.strip()),
    ]
    for iid, kw in zip(ids, kwargs_list):
        cls = inst_dict[iid]
        inst = cls(iid)
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        inst.build_description(**kw)
        if hasattr(inst, "get_instruction_args_keys"):
            args = inst.get_instruction_args() or {}
            if "prompt" in (inst.get_instruction_args_keys() or []):
                inst.build_description(prompt=prompt)
        strict.append(bool(response.strip()) and inst.check_following(response))
        loose.append(any(inst.check_following(v) for v in loose_variants if v.strip()))
    return all(strict), all(loose), strict, loose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True,
                    help="lm-eval IFEval samples_*.jsonl (globs ok)")
    args = ap.parse_args()

    inst_dict = load_instruction_registry()
    files = [f for pat in args.samples for f in glob.glob(pat)]
    if not files:
        raise SystemExit(f"no sample files matched {args.samples}")

    p_strict = p_loose = 0
    i_strict = i_loose = 0
    n_prompt = n_inst = 0
    for f in files:
        for line in open(f):
            s = json.loads(line)
            doc = get_doc(s)
            resp = strip_think(get_resp(s))
            ps, pl, si, li = check_one(inst_dict, doc, resp)
            p_strict += ps
            p_loose += pl
            i_strict += sum(si)
            i_loose += sum(li)
            n_prompt += 1
            n_inst += len(si)

    print(json.dumps({
        "n_prompts": n_prompt, "n_instructions": n_inst,
        "prompt_level_strict_acc": round(p_strict / max(n_prompt, 1), 4),
        "prompt_level_loose_acc": round(p_loose / max(n_prompt, 1), 4),
        "inst_level_strict_acc": round(i_strict / max(n_inst, 1), 4),
        "inst_level_loose_acc": round(i_loose / max(n_inst, 1), 4),
        "note": "think-stripped re-score (rescore_ifeval.py)",
    }, indent=2))


if __name__ == "__main__":
    main()
