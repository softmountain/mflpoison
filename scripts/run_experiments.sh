#!/usr/bin/env bash
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: $0 GPU:CONFIG:SEED [GPU:CONFIG:SEED ...]"
  echo "example: $0 0:configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml:42"
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
results_root="${RESULTS_ROOT:-results}"
monitor_interval="${MONITOR_INTERVAL:-30}"
batch_time="$(date +%H-%M-%S)"
batch_dir="$results_root/batches/$(date +%F)/$batch_time"
status_file="$batch_dir/status.tsv"
mkdir -p "$batch_dir"

declare -a names gpus configs seeds run_dirs pids statuses

write_status() {
  {
    printf "experiment\tgpu\tseed\tpid\tstatus\tconfig\trun_dir\n"
    for i in "${!names[@]}"; do
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${names[$i]}" "${gpus[$i]}" "${seeds[$i]}" "${pids[$i]}" \
        "${statuses[$i]}" "${configs[$i]}" "${run_dirs[$i]}"
    done
  } > "$status_file"
}

for job in "$@"; do
  gpu="${job%%:*}"
  remainder="${job#*:}"
  config="${remainder%%:*}"
  if [ "$remainder" = "$config" ]; then
    echo "missing seed in job: $job" >&2
    exit 2
  fi
  seed="${remainder#*:}"

  name="$(basename "${config%.*}")"
  run_time="$(date +%H-%M-%S)"
  run_dir="$results_root/$(date +%F)/$name/$run_time"
  run_dir="${run_dir}_seed-${seed}"
  mkdir -p "$run_dir"

  command=(
    "$python_bin" -m mflpoison.runner
    --config "$config"
    --run-dir "$run_dir"
  )
  command+=(--seed "$seed")

  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$run_dir/train.log" 2>&1 &
  pid=$!
  names+=("$name")
  gpus+=("$gpu")
  configs+=("$config")
  seeds+=("$seed")
  run_dirs+=("$run_dir")
  pids+=("$pid")
  statuses+=("running")
  echo "started $name on GPU $gpu: pid=$pid, run_dir=$run_dir"
done

write_status

while true; do
  running=0
  for i in "${!pids[@]}"; do
    if [ "${statuses[$i]}" != "running" ]; then
      continue
    fi
    if kill -0 "${pids[$i]}" 2>/dev/null; then
      running=$((running + 1))
      continue
    fi
    if wait "${pids[$i]}"; then
      statuses[$i]="completed"
    else
      statuses[$i]="failed"
    fi
    echo "${names[$i]}: ${statuses[$i]} (${run_dirs[$i]})"
  done
  write_status
  if [ "$running" -eq 0 ]; then
    break
  fi
  sleep "$monitor_interval"
done

echo "batch finished; status: $status_file"
