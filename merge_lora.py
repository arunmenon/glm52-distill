#!/usr/bin/env python3
"""
merge_lora.py — merge a LoRA adapter into its base model (single process).

04_train_kd.py saves only the adapter when --lora is used, because merging
inside a DeepSpeed ZeRO-3 run operates on sharded parameters and corrupts the
merged weights. Run this afterwards to produce the merged checkpoint that the
vLLM evals load. CPU is fine (needs ~2x model size in RAM); no GPU required.

  python merge_lora.py --base /models/glm-4.5-air \
      --adapter ckpts/air_kd/adapter --out ckpts/air_kd/final
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)
    print(f"merged {args.base} + {args.adapter} -> {args.out}")


if __name__ == "__main__":
    main()
