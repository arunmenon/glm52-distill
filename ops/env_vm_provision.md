# Env VM provisioning (Vast KVM, x86)

Checklist distilled from the WALK-era incidents. Every line earned its place.

## Instance selection
- KVM **VM** offer (not container): container instances' `authorized_keys`
  are agent-managed and revert; VMs keep appended keys.
- x86_64, >= 8 cores, >= 150 GB disk (task images are 3-8 GB each and GC'd
  after seal by the worker).
- Named geolocation only (e.g. "TX, US"). Offers geolocating as bare ", US"
  were broken hosts twice (no HF connectivity, dead xet).
- Record the instance id in the journal. NEVER touch instances this project
  did not create — the account runs the user's own experiments too.

## Provisioning steps
1. `echo 'DOCKER_API_VERSION=1.43' >> /etc/environment` then re-login —
   apt's docker client is newer than the image's daemon; pulls fail with
   "client version too new" otherwise.
2. `python3 -m venv /root/venv && /root/venv/bin/pip install pandas pyarrow
   "huggingface_hub[cli]" mini-swe-agent==2.4.6 pyyaml`
3. Copy repo scripts to `/root/exp/`: 09_rehearsal.py, 09_generate_
   trajectories.py, 09d_walk_bugfix.py, 09e_expansion_bugfix.py,
   09c_rescore_logprobs.py, 10e_gold_smoke.py, trajectory_decontam.py,
   ops/env_vm_launch.sh, plus runs/expansion/bugfix/ (queue + gate files).
4. Write `/root/exp/.env.hf`:
   `export PILOT_STORE=ledzepu2/glm52-pilot-artifacts` and
   `export HF_TOKEN=<fine-grained WRITE token scoped to that dataset>`.
   Never ship the laptop's OAuth login token — it refreshes out from under
   a long-running box.
5. Gold smoke first (no teacher needed):
   `nohup /root/venv/bin/python -u 10e_gold_smoke.py >> smoke.log 2>&1 &`
6. Generation (after teacher box is up and tunnel established):
   `export TEACHER_BASE_URL=http://localhost:8000/v1/chat/completions
    TEACHER_API_KEY=... && bash ops/env_vm_launch.sh <run_list.json> 3`

## Operational rules
- All remote ops as script files scp'd over, never inline `ssh "pkill ..."`
  — pgrep/pkill match the ssh client's own argv (killed our workers 4x).
- Workers carry inert argv markers (`xpartN_marker`) for targeting.
- Never `exec -a` to rename processes — it breaks CPython venv discovery.
- `python -u` + per-task ledgers as ground truth; stdout block-buffers
  under nohup.
- Sealed results must leave the box within minutes (sync-on-seal); stopped
  Vast VMs are unreachable until they boot and hosts can reclaim at any
  moment (44 ledgers / $66.73 lost that way once).
