# glm52-distill — End-to-End GLM-5.2 Distillation Pipeline

> **Multi-VM users (8+8+2 GPU topology): start with [README_GCP.md](README_GCP.md).**
> `run_all.sh` and this run-order section are the LEGACY single-node (16-GPU) path.

Time-sliced distillation of **GLM-5.2 (744B MoE, FP8)** into two students on
**16× RTX Pro 6000 Blackwell (96 GB)**:

- **Student A:** GLM-family small model (tokenizer-matched) → SFT **+ top-K logit KD**
- **Student B:** Qwen3-8B (tokenizer-mismatched) → sequence-level KD (SFT on teacher outputs)

Phase 1 uses all 16 GPUs to serve the teacher and generate a corpus with stored
top-20 logprobs. Phase 2 tears the teacher down and uses all 16 GPUs for training.

Everything is deterministic: seed 42 throughout, per-sample generation seeds,
pinned environments, SHA256 manifests at every stage.

---

## Repo layout

```
00_setup.sh                    env creation, weight download, disk checks
00_check_tokenizer.py          HARD GATE: decides if Student A gets logit KD
01_build_prompts.py            build 150k prompt corpus + 1k held-out set (+ MANIFEST)
10_serve_teacher.sh            launch GLM-5.2 FP8 on all 16 GPUs (vLLM, single node)
11_ray_cluster.sh              bring up the 2-node Ray cluster (multi-VM topology)
12_serve_teacher_multinode.sh  teacher across VM-A + VM-B (TP=8 x PP=2 over Ray)
02_generate.py                 async corpus generation with top-20 logprob capture
03_pack_dataset.py             align teacher tokens/logprobs to student inputs, save HF datasets
04_train_kd.py                 unified trainer: alpha=0 → plain SFT; alpha>0 → SFT + logit KD; --lora supported
merge_lora.py                  merge LoRA adapter into base (separate step; ZeRO-3-safe)
05_eval_gen.py                 generate student answers on the held-out set (offline vLLM)
06_judge.py                    pairwise student-vs-teacher judging via teacher endpoint
07_eval_benchmarks.sh          lm-eval-harness benchmarks (IFEval, MMLU-Pro, GSM8K, ...)
20_sync.sh                     GCS push/pull for local-disk-only VMs
run_all.sh                     the whole pipeline in order, with phase gates (single node)
run_vm_a.sh / _b.sh / _c.sh    per-VM orchestration for the 8+8+2 GCP topology
common.sh                      shared helpers (serve/wait/smoke, pod env, artifact store)
pilot_runpod.sh                single-GPU end-to-end rehearsal (Qwen3-8B as teacher stand-in)
pilot_teacher.sh               leg 1: real GLM-5.2 serving + corpus (8x H200 or 6x B200)
pilot_student.sh               leg 2: real GLM-4.5-Air LoRA KD + merge (4x RTX PRO 6000 / H100)
configs/ds_zero3.yaml          accelerate + DeepSpeed ZeRO-3 config (16 GPUs; _8gpu/_4gpu/_1gpu variants)
```

## Exact run order

```bash
# ---- Day 0: setup ----------------------------------------------------------
bash 00_setup.sh                          # envs, downloads, disk check
source ~/venvs/train/bin/activate
python 00_check_tokenizer.py \
  --teacher /models/glm-5.2-fp8 --student-a /models/glm-4.5-air
# Prints LOGIT_KD_OK=1 or 0. If 0, Student A is trained SFT-only (alpha=0).

python 01_build_prompts.py --out data/    # -> data/prompts.jsonl, data/prompts_heldout.jsonl

# ---- Days 1-4: Phase 1 (all 16 GPUs = teacher) ------------------------------
bash 10_serve_teacher.sh                  # wait for "Application startup complete"
source ~/venvs/serve/bin/activate
python 02_generate.py --prompts data/prompts.jsonl --out corpus/ \
  --base-url http://localhost:8000/v1 --model glm-5.2 --concurrency 112 \
  --best-of 4        # rejection sampling: 4 candidates/prompt, teacher-judged best.
                     # ~4x generation cost -> pair with `01_build_prompts.py --n-total 100000`
                     # or drop the flag (default 1) for the cheap single-sample corpus.
# resumable; rerun the same command after any crash
pkill -f "vllm serve"                     # tear down teacher

# ---- Days 4-8: Phase 2 (all 16 GPUs = training) ------------------------------
source ~/venvs/train/bin/activate
python 03_pack_dataset.py --corpus corpus/ --heldout data/prompts_heldout.jsonl \
  --teacher /models/glm-5.2-fp8 --student-a /models/glm-4.5-air \
  --student-b /models/qwen3-8b --out packed/

# Run 1a: Student B plain SFT (baseline)
accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
  --model /models/qwen3-8b --data packed/student_b --out ckpts/qwen3_sft --alpha 0.0

# Run 1b: Student A plain SFT baseline (LoRA; adapter is merged in a separate step)
accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
  --model /models/glm-4.5-air --data packed/student_a --out ckpts/air_sft \
  --alpha 0.0 --lora --lr 1e-4
python merge_lora.py --base /models/glm-4.5-air --adapter ckpts/air_sft/adapter --out ckpts/air_sft/final

# Run 2: Student A SFT + logit KD (only if LOGIT_KD_OK=1)
accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
  --model /models/glm-4.5-air --data packed/student_a --out ckpts/air_kd \
  --alpha 0.5 --temperature 2.0 --lora --lr 1e-4
python merge_lora.py --base /models/glm-4.5-air --adapter ckpts/air_kd/adapter --out ckpts/air_kd/final

# ---- Days 8-10: evaluation ---------------------------------------------------
python 05_eval_gen.py --model ckpts/qwen3_sft/final --heldout data/prompts_heldout.jsonl --out evals/qwen3_sft.jsonl
python 05_eval_gen.py --model ckpts/air_sft/final  --heldout data/prompts_heldout.jsonl --out evals/air_sft.jsonl
python 05_eval_gen.py --model ckpts/air_kd/final   --heldout data/prompts_heldout.jsonl --out evals/air_kd.jsonl

bash 10_serve_teacher.sh                  # re-host teacher as judge + reference
python 05_eval_gen.py --served --base-url http://localhost:8000/v1 --model glm-5.2 \
  --heldout data/prompts_heldout.jsonl --out evals/teacher_ref.jsonl
python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/air_kd.jsonl  --out evals/judge_air_kd.json
python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/air_sft.jsonl --out evals/judge_air_sft.json
python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/qwen3_sft.jsonl --out evals/judge_qwen3.json
pkill -f "vllm serve"

bash 07_eval_benchmarks.sh ckpts/qwen3_sft/final ckpts/air_sft/final ckpts/air_kd/final
```

Or simply: `bash run_all.sh` (runs the above with gates; edit paths at the top).

## Design decisions (so there are no surprises)

1. **Token-exact alignment.** Student A inputs are built as
   `chat_template_prefix_ids + teacher_response_token_ids` — the teacher's raw
   generated token ids are reused directly (valid because the tokenizer gate
   passed). This makes the stored top-20 logprobs align 1:1 with labels with
   zero re-tokenization drift.
2. **No reasoning parser on the server.** `10_serve_teacher.sh` deliberately does
   NOT enable `--reasoning-parser`, so thinking traces stay in the token stream
   and remain part of the distillation signal. Students learn to think.
3. **No sequence packing.** Samples are truncated at 8192 and padded per batch.
   Packing with per-position teacher arrays is error-prone; we trade ~15%
   throughput for correctness.
4. **One trainer for everything.** `--alpha 0.0` = pure SFT (works for both
   students, no top-K columns needed); `--alpha 0.5` = SFT + logit KD.
5. **Truncated generations are dropped** (finish_reason != "stop") — training on
   cut-off reasoning teaches bad habits. With `--best-of N`, truncated candidates
   are discarded first and the teacher judges among the survivors (temperature-0,
   seeded, single-token verdict), so the kept sample is both complete and best-of-N.
   `bon_n`/`bon_pick` are stored per row for auditability.
6. **Held-out contamination is asserted, not hoped:** `03_pack_dataset.py` hard-fails
   if any held-out prompt hash appears in training data.

## Success criteria

- Student A (KD) retains ≥ 75% of teacher's held-out win-rate-adjusted score and
  beats Student A (SFT) on ≥ 2 of {judge win-rate, GSM8K, IFEval}.
- If logit KD does not beat SFT: check `loss_kl` in W&B — if it collapses to ~0
  early, lower alpha to 0.3; if it dominates, raise temperature to 3.0.

## Failure modes / rollback

- vLLM can't load GLM-5.2 → `pip install -U vllm --pre` or SGLang; last resort:
  swap teacher to GLM-4.7 (355B, fits on 8 GPUs; same scripts, `--tensor-parallel-size 8`, no PP).
- Serving OOM → in `10_serve_teacher.sh`, set `--max-model-len 16384 --max-num-seqs 64`.
- Generation slower than ~1.5k tok/s aggregate → cut `01_build_prompts.py --n-total` to 100000.
- Training OOM (Air LoRA) → micro_bsz stays 1, raise `--grad-accum` to 16 and
  re-pack with `03_pack_dataset.py --max-len 6144`.
