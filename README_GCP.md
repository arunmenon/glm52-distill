# GCP Topology Guide — 3 VMs (8 + 8 + 2 RTX Pro 6000), local disk only

## QUICKSTART (copy-paste per VM)

```bash
# --- ALL THREE VMs, once ---
sudo apt-get update && sudo apt-get install -y python3-venv zip curl git
gcloud auth login && gcloud config set project <YOUR_PROJECT>       # or rely on VM service account
export GCS_BUCKET=gs://<bucket>/glm52-distill
# optional but recommended:
#   ~/venvs/*/bin/hf auth login       (only if any model repo is gated)
#   wandb login <key>                  (or: export WANDB_MODE=offline)

# --- VM-A and VM-B (8 GPUs each) ---
EXPECTED_GPUS=8 bash 00_setup.sh                 # full downloads (~1TB, budget 1-3h)

# --- VM-C (2 GPUs) ---
EXPECTED_GPUS=2 SKIP_TEACHER=1 bash 00_setup.sh  # tokenizers/configs only + qwen3-8b

# --- Launch (three terminals) ---
# VM-A:  bash run_vm_a.sh
# VM-B:  HEAD_IP=<vm-a-internal-ip> bash run_vm_b.sh          (when A prints the IP)
# VM-C:  PILOT=1 TEACHER_URL=http://<vm-a-ip>:8000/v1 bash run_vm_c.sh   (pilot first!)
# then rerun VM-C without PILOT=1 for the full corpus.
```

The scripts pause and tell you at each hand-off point. Every phase pushes its
outputs to $GCS_BUCKET, so a VM crash or preemption never loses more than the
in-flight phase.


This supersedes `run_all.sh` (single-node) for your topology. The single-node
script still works for the 2-GPU pilot of small pieces, but the real run uses
the three per-VM scripts.

## Role assignment

| VM | GPUs | Phase 1 (days 1-5) | Phase 2 (days 5-8) | Phase 3 (days 8-10) |
|----|------|--------------------|--------------------|---------------------|
| A  | 8    | Ray head, teacher PP stage 1 | Train Qwen3-8B (full FT) | lm-eval benchmarks, teacher re-host stage 1 |
| B  | 8    | Ray worker, teacher PP stage 2 | Train GLM-4.5-Air (LoRA, SFT + KD) | Air held-out gens, teacher re-host stage 2 |
| C  | 2    | Driver: prompts, gate, 02_generate client | idle / W&B monitoring | Qwen gens (tp=2), judging client |

Both students train **in parallel** (one per 8-GPU VM) — faster than the
original single-node plan.

## One-time GCP prerequisites

1. **Same zone** for all three VMs (us-south1); add a compact placement policy
   for A+B if possible.
2. **gVNIC + Tier_1 networking** on VM-A and VM-B (8-GPU G4 shapes support up
   to ~400 Gbps internal). PP=2 only ships activations cross-VM, so even a
   fraction of that is fine.
3. **Firewall rules** between A and B internal IPs: tcp 6379 (Ray), tcp
   10000-19999 (Ray workers), and NCCL's dynamic TCP range (simplest: allow all
   TCP between the two internal IPs). Also tcp 8000 from VM-C to VM-A.
4. **GCS bucket** in the same region: `gsutil mb -l us-south1 gs://<bucket>`.
   Every VM: `export GCS_BUCKET=gs://<bucket>/glm52-distill`. VM service
   accounts need Storage Object Admin on the bucket.
5. **Local SSD is ephemeral.** Anything not pushed to GCS is lost on
   stop/preemption. The run scripts push after every phase; don't remove those
   lines. If these are Spot VMs, also raise `save_steps` pushes: add a cron
   `bash 20_sync.sh push ckpts ckpts_wip` every 30 min during training.
6. **Weights are downloaded per-VM** (no shared FS): run `00_setup.sh` on A and
   B in full; on C you can skip the teacher download *except* the tokenizer
   files, or just download everything if disk allows. The teacher weights must
   sit at the same path (/models/glm-5.2-fp8) on A and B.

## Run order

```bash
# every VM, once:
export GCS_BUCKET=gs://<bucket>/glm52-distill
bash 00_setup.sh

# VM-A:                          bash run_vm_a.sh        (starts Ray head + teacher)
# VM-B (when A prints the IP):   HEAD_IP=<a-ip> bash run_vm_b.sh
# VM-C (when teacher is READY):  TEACHER_URL=http://<a-ip>:8000/v1 bash run_vm_c.sh
# ...scripts prompt you at the two hand-off points (teacher up / corpus done).
```

Pilot first: on VM-C run with `PILOT=1` (500 prompts) once the teacher is up —
it exercises prompt building, the tokenizer gate, and generation against the
real teacher. (Packing/training/judging are rehearsed separately by
`pilot_runpod.sh` on any single GPU.) After the pilot, clear state before the
real run: `rm -rf .state data corpus gate evals` on VM-C plus the matching
GCS paths — the scripts refuse to start if pilot state is still present.

## Throughput reality check for this topology

- PP across VMs adds pipeline-bubble + network latency: plan for **1-2k tok/s
  aggregate** decode. With BEST_OF=4 the default corpus is therefore set to
  100k prompts in `run_vm_c.sh` (~5 days worst case). Measure in the pilot and
  resize.
- If multi-node serving proves flaky, the documented fallback order is:
  (1) W4A16-quantize the teacher so it fits on ONE 8-GPU VM (~390 GB),
  (2) swap teacher to GLM-4.7 FP8 (355 GB, single VM, zero code changes).
  Both fallbacks also free VM-B to start training earlier.

## Cost note

Three G4 VMs idle-burning is real money; the scripts are phased so you can
stop VM-B entirely during Phase 3a and stop VM-C during Phase 2. Nothing is
lost as long as the phase-end sync ran (check `gcloud storage ls $GCS_BUCKET`).
