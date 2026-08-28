#!/bin/bash
# teacher_box_launch.sh — serve Qwen3.8-27B-FP8 on the GPU box + reverse
# tunnel to the env VM. Run ON the teacher box.
#
# Usage:
#   export TEACHER_API_KEY=<random secret shared with the env VM>
#   export ENV_VM_SSH="-p <port> root@<ip>"   # env VM ssh coordinates
#   bash teacher_box_launch.sh
#
# EVERY serve flag below is load-bearing (each was discovered by a failure):
#   --enable-auto-tool-choice      absent -> HTTP 400 on every agent request
#   --tool-call-parser qwen3_xml   hermes leaves <function=bash> XML in
#                                  content; Qwen3.8 emits XML tool calls
#   --reasoning-parser qwen3       absent -> think-blocks leak into content,
#                                  breaking trajectory format + 09c re-render
#   --max-num-seqs 16              hybrid Mamba layers need a cache block per
#                                  slot; large values fail engine init
#   --served-model-name            must equal the runner's MODEL_NAME so the
#                                  TEACHER_BASE_URL shim needs no rename
#
# The reverse tunnel (-R) exists because Vast does not map extra container
# ports on raw create: the GPU box pushes :8000 to the env VM's localhost.
set -euo pipefail
: "${TEACHER_API_KEY:?}" "${ENV_VM_SSH:?}"
exec > /root/launch_teacher.out 2>&1

source /venv/main/bin/activate
nohup vllm serve Qwen/Qwen3.8-27B-FP8 \
    --served-model-name qwen/qwen3.8-27b \
    --api-key "$TEACHER_API_KEY" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    > /root/serve.log 2>&1 &

cat > /root/rtunnel.sh <<EOF
#!/bin/bash
while true; do
  ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes \\
      -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \\
      -N -R 8000:localhost:8000 $ENV_VM_SSH
  sleep 8
done
EOF
chmod +x /root/rtunnel.sh
nohup bash /root/rtunnel.sh > /root/rtunnel.log 2>&1 &

cat > /root/serve_watchdog.sh <<'EOF'
#!/bin/bash
# restart vllm if it dies (OOM, engine crash); tunnel loop self-heals
while true; do
  sleep 120
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "[serve_watchdog] $(date) vllm dead - relaunching" >> /root/serve_watchdog.log
    source /venv/main/bin/activate
    nohup vllm serve Qwen/Qwen3.8-27B-FP8 \
        --served-model-name qwen/qwen3.8-27b \
        --api-key "$TEACHER_API_KEY" \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 131072 --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 >> /root/serve.log 2>&1 &
  fi
done
EOF
chmod +x /root/serve_watchdog.sh
TEACHER_API_KEY="$TEACHER_API_KEY" nohup bash /root/serve_watchdog.sh >/dev/null 2>&1 &
disown -a
sleep 8
echo "vllm procs: $(pgrep -c -f 'vllm serve') | tunnel: $(pgrep -c -f rtunnel)"
echo LAUNCHED
