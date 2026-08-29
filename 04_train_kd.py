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
    import glob as _glob
    last_ckpt = bool(_glob.glob(os.path.join(args.out, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=last_ckpt or None)

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
