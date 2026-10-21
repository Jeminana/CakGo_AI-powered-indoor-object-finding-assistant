#!/usr/bin/env bash
# Runs the zero-shot comparison on one Colab VM, one Python process per run, in a fixed order.
# Started in the background through the Colab CLI (see README.md). Progress: /content/zs_status.txt
# (current step), /content/zs_done.txt (finished runs), /content/zs_log.txt (full output).
#
# Usage: bash tests/zero_shot/colab_job.sh [RUN_KEY ...]   (default: every run in ORDER)
set -u
REPO=/content/repo
WORK=/content/work  # YOLOE weights and the MobileCLIP2 text encoder download here, not into the repo
mkdir -p "$WORK"
cd "$WORK" || exit 1

declare -A RUNS=(
  [gdino_multi]="--model gdino --prompt-mode multi --run-name t4_gdino-tiny_multi"
  [yoloe26s_single]="--model yoloe --weights $WORK/yoloe-26s-seg.pt --prompt-mode single --run-name t4_yoloe-26s_single"
  [yoloe26s_multi]="--model yoloe --weights $WORK/yoloe-26s-seg.pt --prompt-mode multi --run-name t4_yoloe-26s_multi"
  [yoloe26x_single]="--model yoloe --weights $WORK/yoloe-26x-seg.pt --prompt-mode single --run-name t4_yoloe-26x_single"
  [yoloe26x_multi]="--model yoloe --weights $WORK/yoloe-26x-seg.pt --prompt-mode multi --run-name t4_yoloe-26x_multi"
  [sam3_single]="--model sam3 --prompt-mode single --run-name t4_sam3_single"
  [sam3_multi]="--model sam3 --prompt-mode multi --run-name t4_sam3_multi"
)
ORDER=(gdino_multi yoloe26s_single yoloe26s_multi yoloe26x_single yoloe26x_multi sam3_single sam3_multi)

keys=("$@")
[ ${#keys[@]} -eq 0 ] && keys=("${ORDER[@]}")
for key in "${keys[@]}"; do
  echo "running $key since $(date -u +%H:%M:%S) UTC" > /content/zs_status.txt
  echo "===== $key $(date -u)" >> /content/zs_log.txt
  # SAM 3 is a gated checkpoint: it is copied into the VM's Hugging Face cache beforehand (see README.md),
  # so its runs read only that cache and never need a token on the VM.
  offline=0
  [[ $key == sam3* ]] && offline=1
  # shellcheck disable=SC2086  # RUNS values are word-split on purpose
  if HF_HUB_OFFLINE=$offline python "$REPO/tests/zero_shot/run_zero_shot.py" ${RUNS[$key]} --device cuda \
      >> /content/zs_log.txt 2>&1; then
    echo "$key ok" >> /content/zs_done.txt
  else
    echo "$key FAILED" >> /content/zs_done.txt
  fi
done
echo "idle since $(date -u +%H:%M:%S) UTC" > /content/zs_status.txt
