#!/usr/bin/env python3
"""
04_train_kd.py — Unified student trainer.

  --alpha 0.0            -> plain SFT (sequence-level KD). Works on datasets
                            with or without top-K columns (student_b has none).
  --alpha 0.5 --temperature 2.0
                         -> SFT + top-K logit distillation:
       L = (1-a)*CE(student, teacher_token) + a*T^2*KL(p_teacher_topK || p_student_topK)
     Both distributions restricted to the teacher's stored top-K ids,
     temperature-scaled, renormalized over K. Prompt positions masked.
  --lora                 -> LoRA r=128, alpha=256 on all linear layers
                            (use for GLM-4.5-Air-class students).

Dataset contract (from 03_pack_dataset.py): per example
  input_ids [L], labels [L] (-100 on prompt), and for logit-KD datasets
  topk_ids [L,K] (-1 pad), topk_logprobs [L,K] (-1e9 pad), where index i
  describes the teacher distribution over labels[i].

Launch:
  accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
      --model /models/qwen3-8b --data packed/student_b --out ckpts/qwen3_sft --alpha 0.0
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

IGNORE = -100
K = 20


class KDCollator:
    def __init__(self, pad_token_id: int, has_topk: bool):
        self.pad = pad_token_id
        self.has_topk = has_topk

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        ids, att, lab, tids, tlps = [], [], [], [], []
        for f in features:
            n = len(f["input_ids"])
            p = max_len - n
            ids.append(list(f["input_ids"]) + [self.pad] * p)
            att.append([1] * n + [0] * p)
            lab.append(list(f["labels"]) + [IGNORE] * p)
            if self.has_topk:
                tids.append([list(x) for x in f["topk_ids"]] + [[-1] * K] * p)
                tlps.append([list(x) for x in f["topk_logprobs"]] + [[-1e9] * K] * p)
        batch = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
        }
        if self.has_topk:
            batch["topk_ids"] = torch.tensor(tids, dtype=torch.long)
            batch["topk_logprobs"] = torch.tensor(tlps, dtype=torch.float32)
        return batch


class KDTrainer(Trainer):
    def __init__(self, *args, alpha: float = 0.0, kd_temperature: float = 2.0, **kw):
        super().__init__(*args, **kw)
        self.alpha = alpha
        self.T = kd_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        topk_ids = inputs.pop("topk_ids", None)
        topk_lps = inputs.pop("topk_logprobs", None)
        labels = inputs["labels"]

        if self.alpha == 0.0 or topk_ids is None:
            # SFT path: let the model compute CE internally so liger's fused
            # linear CE applies — materializing logits for an 80k-token
            # sequence over a 248k vocab is a 40GB tensor and OOMs any 48GB
            # card. The KD path below still needs logits; long-sequence KD
            # requires a chunked gather (future work).
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            self._log_terms(loss, torch.tensor(0.0))
            return (loss, outputs) if return_outputs else loss

        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits

        # Shift: logits at position i predict token i+1.
        s_logits = logits[:, :-1, :]
        s_labels = labels[:, 1:]
        mask = s_labels.ne(IGNORE)
        n_tok = mask.sum().clamp(min=1)

        ce = F.cross_entropy(
            s_logits.reshape(-1, s_logits.size(-1)).float(),
            s_labels.reshape(-1),
            ignore_index=IGNORE,
            reduction="sum",
        ) / n_tok

        t_ids = topk_ids[:, 1:, :]
        t_lps = topk_lps[:, 1:, :]
        safe_ids = t_ids.clamp(min=0)
        s_topk = torch.gather(s_logits.float(), -1, safe_ids)
        pad_k = t_ids.eq(-1)
        s_topk = s_topk.masked_fill(pad_k, -1e9)
        t_topk = t_lps.masked_fill(pad_k, -1e9)

        log_ps = F.log_softmax(s_topk / self.T, dim=-1)
        log_pt = F.log_softmax(t_topk / self.T, dim=-1)
        pt = log_pt.exp()
        kl_pos = (pt * (log_pt - log_ps)).sum(-1)
        # exclude positions with no teacher dist at all (e.g. appended eos)
        valid = mask & (~pad_k.all(-1))
        kl = (kl_pos * valid).sum() / valid.sum().clamp(min=1)

        loss = (1.0 - self.alpha) * ce + self.alpha * (self.T**2) * kl
        self._log_terms(ce, kl)
        return (loss, outputs) if return_outputs else loss

    def _log_terms(self, ce, kl):
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({"loss_ce": float(ce.detach()), "loss_kl": float(kl.detach())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro-bsz", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="proxy rung: train on the first N samples only (0=all)")
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="drop rows longer than this (0=none); checkpointed "
                         "layer inputs alone are ~0.4MB/token on a 9B — "
                         "sequences past ~65k cannot fit a 48GB card")
    ap.add_argument("--mix-data", default="",
                    help="hub id of a general instruct dataset (chat 'messages'"
                         " column) to interleave against format collapse — "
                         "44 same-format trajectories cost v0 ~12 IFEval pts")
    ap.add_argument("--mix-ratio", type=float, default=0.25,
                    help="fraction of train rows drawn from --mix-data")
    ap.add_argument("--mix-data-revision", default=None,
                    help="pin the mix dataset revision (trial-id integrity)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_from_disk(args.data)
    if args.max_seq_len:
        before = {k: len(ds[k]) for k in ds}
        ds = ds.filter(lambda r: len(r["input_ids"]) <= args.max_seq_len)
        for k in ds:
            if len(ds[k]) != before[k]:
                print(f"max-seq-len {args.max_seq_len}: {k} "
                      f"{before[k]} -> {len(ds[k])} rows")
    if args.max_train_samples and len(ds["train"]) > args.max_train_samples:
        ds["train"] = ds["train"].select(range(args.max_train_samples))
        print(f"proxy rung: train limited to {args.max_train_samples} samples")

    if args.mix_ratio and not (0 < args.mix_ratio < 1) and args.mix_data:
        sys.exit(f"--mix-ratio must be in (0,1), got {args.mix_ratio}")
    if args.mix_data and args.mix_ratio > 0:
        # tokenize general chat rows through the SAME template/mask rules as
        # trajectories (assistant turns trainable) so the mix differs only
        # in content, never in convention
        import random as _random
        from datasets import Dataset, concatenate_datasets, load_dataset
        n_traj = len(ds["train"])
        n_mix = int(round(n_traj * args.mix_ratio / (1 - args.mix_ratio)))
        if has_topk_probe := ("topk_ids" in ds["train"].column_names):
            sys.exit("--mix-data unsupported on top-K KD packs: mix rows "
                     "would carry None topk arrays into the collator")
        # streaming: the pinned Tulu mixture is ~7.2GB / 939k rows on
        # disk if materialized; we need a few dozen rows (r3 f10)
        general = load_dataset(args.mix_data, split="train",
                               revision=args.mix_data_revision,
                               streaming=True)
        general = general.shuffle(seed=args.seed, buffer_size=10000)
        # Label construction reuses 03c's segment renderer — the same
        # production path that masks the trajectory corpus (assistant turns
        # trainable, everything else IGNORE). The earlier prefix-diff
        # approach corrupted spans: Qwen3.5's template rewrites historical
        # assistant turns, so prefix lengths are not stable (review r3 b3).
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "mt", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "03c_build_multiturn_dataset.py"))
        _mt = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mt)

        mix_rows, rejected = [], {"no_messages": 0, "render": 0,
                                  "too_long": 0, "no_trainable": 0}
        for ex in general:
            msgs = ex.get("messages") or ex.get("conversations")
            if not msgs:
                rejected["no_messages"] += 1
                continue
            try:
                segments = _mt.render_qwen3_segments(list(msgs))
                ids, labels = _mt.tokenize_with_mask(
                    tok, segments, args.max_seq_len or 32768)
            except Exception:
                rejected["render"] += 1
                continue
            if ids is None:
                rejected["too_long"] += 1
                continue
            if all(l == IGNORE for l in labels):
                rejected["no_trainable"] += 1
                continue
            mix_rows.append({"input_ids": list(ids), "labels": list(labels)})
            if len(mix_rows) >= n_mix:
                break
        if len(mix_rows) < n_mix:
            sys.exit(f"mix-data underfilled: {len(mix_rows)}/{n_mix} "
                     f"usable rows (rejected: {rejected})")
        print(f"mix-data: +{len(mix_rows)} general rows "
              f"({args.mix_ratio:.0%} target) -> train={len(ds['train'])} "
              f"rejected={rejected}")

    has_topk = "topk_ids" in ds["train"].column_names

    # W&B only when actually configured; otherwise a headless VM hangs at the
    # interactive wandb login prompt.
    if os.environ.get("WANDB_MODE") == "disabled":
        report_to = "none"
    elif os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE"):
        report_to = "wandb"
    else:
        report_to = "none"

    targs_kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.micro_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        # the trajectory corpus can be as few as 10 optimizer steps; any
        # save interval above that loses the whole run to a post-train crash
        save_steps=5,
        save_total_limit=2,
        # ZeRO-3 checkpoints include ~12x-model-size fp32 optimizer state
        # (117GB for a 9B) — one checkpoint filled a 150GB disk and ENOSPC'd
        # the final save. Weights-only checkpoints lose optimizer resume but
        # keep the run's artifacts deliverable.
        save_only_model=True,
        # in-Trainer eval OOMs: eval forwards run without gradient
        # checkpointing, and a 63k-token sdpa attention materializes ~32GB.
        # The transfer screen (05-07) is the real eval; skip Trainer eval.
        eval_strategy="no",
        max_grad_norm=1.0,
        report_to=report_to,
        run_name=f"{args.out.split('/')[-1]}_a{args.alpha}_T{args.temperature}",
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # 80k-token sequences x 248k vocab = 30-40GB materialized logits;
        # liger's fused linear CE computes loss without materializing them
        use_liger_kernel=True,
    )
    # transformers v5 ships breaking TrainingArguments changes in minor
    # releases (warmup_ratio folded into warmup_steps-as-float in 5.0);
    # filter to the installed version's signature so both majors work
    import inspect
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" not in valid and "warmup_steps" in valid:
        targs_kwargs["warmup_steps"] = targs_kwargs.pop("warmup_ratio")
    for k in sorted(set(targs_kwargs) - valid):
        print(f"targs: dropping {k} (absent in this transformers version)")
        targs_kwargs.pop(k)
    targs = TrainingArguments(**targs_kwargs)

    # Model MUST load after TrainingArguments: transformers only shards at load
    # (zero.Init) once the deepspeed config is registered — loading first left
    # every rank holding the full model + grads (~42GB flat, OOM at any cap).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True,
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.lora:
        from peft import LoraConfig, get_peft_model
        # REQUIRED with gradient checkpointing: without this, inputs carry no
        # grad and LoRA adapters silently receive zero gradient.
        model.enable_input_require_grads()
        lcfg = LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
            # MoE students (GLM-4.5-Air): keep LoRA off the expert routers —
            # adapting them destabilizes routing. Matches `mlp.gate`, not
            # `gate_proj`. Harmless no-op on dense models.
            exclude_modules=["gate"],
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()


    trainer = KDTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=KDCollator(tok.pad_token_id or tok.eos_token_id, has_topk),
        alpha=args.alpha if has_topk else 0.0,
        kd_temperature=args.temperature,
    )
    # resume only from checkpoints that carry optimizer state; weights-only
    # checkpoints (save_only_model) cannot resume exactly — restart clean
    # (autoloop re-review finding 5)
    import glob as _glob
    ckpts = sorted(_glob.glob(os.path.join(args.out, "checkpoint-*")),
                   key=lambda c: int(c.rsplit("-", 1)[-1]))
    last = ckpts[-1] if ckpts else None
    resumable = last and (
        _glob.glob(os.path.join(last, "global_step*"))
        or (os.path.exists(os.path.join(last, "optimizer.pt"))
            and os.path.exists(os.path.join(last, "scheduler.pt"))
            and _glob.glob(os.path.join(last, "rng_state*.pth"))))
    trainer.train(resume_from_checkpoint=last if resumable else None)

    if args.lora:
        # ZeRO-3 shards parameters across ranks; merging inside the training
        # process operates on shard-sized tensors and corrupts the merged
        # weights. Save the adapter here (trainer.save_model gathers it
        # correctly) and merge in a separate single process via merge_lora.py.
        adapter = args.out + "/adapter"
        trainer.save_model(adapter)
        if trainer.is_world_process_zero():
            tok.save_pretrained(adapter)
        print(f"saved LoRA adapter to {adapter}")
        print(f"next: python merge_lora.py --base {args.model} "
              f"--adapter {adapter} --out {args.out}/final")
    else:
        final = args.out + "/final"
        trainer.save_model(final)
        if trainer.is_world_process_zero():
            tok.save_pretrained(final)
        print(f"saved final model to {final}")


if __name__ == "__main__":
    main()
