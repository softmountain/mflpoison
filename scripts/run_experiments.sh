#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
usage: scripts/run_experiments.sh [options] CONFIG:SEED [CONFIG:SEED ...]

options:
  --gpus LIST                    comma-separated GPU pool (default: 0,1,2,3)
  --canonical-clean-config PATH configuration used for M* and clean replicas
  --experiment-branch NAME      attack or defended (default: attack)
  --monitor-interval SECONDS     scheduler polling interval (default: 30)
  --idle-memory-mib MIB          maximum idle GPU memory use (default: 1024)

The scheduler creates this dependency chain for every seed:
  mstar -> clean-1..clean-5 -> canonical-aggregate -> selected branch jobs

Environment overrides:
  PYTHON_BIN, ARTIFACT_ROOT, NVIDIA_SMI_BIN, BATCH_ID, SCHEDULER_LOCK_FILE
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

python_bin="${PYTHON_BIN:-python}"
artifact_root="${ARTIFACT_ROOT:-artifact}"
nvidia_smi_bin="${NVIDIA_SMI_BIN:-nvidia-smi}"
scheduler_lock_file="${SCHEDULER_LOCK_FILE:-/tmp/mflpoison-run-experiments.lock}"
gpu_csv="0,1,2,3"
canonical_config="configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml"
experiment_branch="attack"
monitor_interval="30"
idle_memory_mib="1024"
clean_replicas=5
batch_id="${BATCH_ID:-$(date +%Y%m%d-%H%M%S)}"
declare -a requested_jobs=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus)
      [ "$#" -ge 2 ] || die "--gpus requires a value"
      gpu_csv="$2"
      shift 2
      ;;
    --canonical-clean-config)
      [ "$#" -ge 2 ] || die "--canonical-clean-config requires a path"
      canonical_config="$2"
      shift 2
      ;;
    --experiment-branch)
      [ "$#" -ge 2 ] || die "--experiment-branch requires attack or defended"
      experiment_branch="$2"
      shift 2
      ;;
    --monitor-interval)
      [ "$#" -ge 2 ] || die "--monitor-interval requires seconds"
      monitor_interval="$2"
      shift 2
      ;;
    --idle-memory-mib)
      [ "$#" -ge 2 ] || die "--idle-memory-mib requires MiB"
      idle_memory_mib="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        requested_jobs+=("$1")
        shift
      done
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      requested_jobs+=("$1")
      shift
      ;;
  esac
done

[ "${#requested_jobs[@]}" -gt 0 ] || {
  usage >&2
  exit 2
}
case "$experiment_branch" in
  attack|defended) ;;
  *) die "--experiment-branch must be attack or defended" ;;
esac
[[ "$idle_memory_mib" =~ ^[0-9]+$ ]] || die "--idle-memory-mib must be an integer"
[[ "$monitor_interval" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "--monitor-interval must be a non-negative number"
[[ "$batch_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "BATCH_ID contains unsafe characters"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v realpath >/dev/null 2>&1 || die "realpath is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"
if [[ "$nvidia_smi_bin" == */* ]]; then
  [ -x "$nvidia_smi_bin" ] || die "NVIDIA_SMI_BIN is not executable: $nvidia_smi_bin"
else
  command -v "$nvidia_smi_bin" >/dev/null 2>&1 || die "nvidia-smi is required"
fi

IFS=',' read -r -a gpu_pool <<< "$gpu_csv"
[ "${#gpu_pool[@]}" -gt 0 ] || die "GPU pool cannot be empty"
declare -A requested_gpu=()
for position in "${!gpu_pool[@]}"; do
  gpu="${gpu_pool[$position]//[[:space:]]/}"
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index: ${gpu_pool[$position]}"
  [ -z "${requested_gpu[$gpu]:-}" ] || die "duplicate GPU index: $gpu"
  requested_gpu[$gpu]=1
  gpu_pool[$position]="$gpu"
done

available_gpu_lines="$($nvidia_smi_bin --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)" \
  || die "cannot query GPU indices"
for gpu in "${gpu_pool[@]}"; do
  echo "$available_gpu_lines" | grep -Eq "^[[:space:]]*$gpu[[:space:]]*$" \
    || die "GPU $gpu is not available"
done

repo_root="$(git rev-parse --show-toplevel)" || die "cannot resolve repository root"
repo_root="$(realpath -e -- "$repo_root")" || die "cannot resolve repository root"
experiments_root="$repo_root/configs/experiments"
[ -d "$experiments_root" ] || die "experiment config directory not found: $experiments_root"
experiments_root="$(realpath -e -- "$experiments_root")" \
  || die "cannot resolve experiment config directory"

resolve_experiment_config() {
  local supplied="$1" resolved relative
  [ -f "$supplied" ] || die "experiment config not found: $supplied"
  resolved="$(realpath -e -- "$supplied")" || die "cannot resolve config: $supplied"
  case "$resolved" in
    "$experiments_root"/*) ;;
    *) die "config must be inside $experiments_root: $supplied" ;;
  esac
  relative="${resolved#"$experiments_root"/}"
  [[ "$relative" =~ ^[A-Za-z0-9._/-]+[.](yaml|yml)$ ]] \
    || die "config path contains unsupported characters: $supplied"
  printf '%s' "$resolved"
}

canonical_config="$(resolve_experiment_config "$canonical_config")" || exit $?
lock_parent="$(dirname -- "$scheduler_lock_file")"
[ -d "$lock_parent" ] || die "scheduler lock directory does not exist: $lock_parent"
exec 9>>"$scheduler_lock_file"
flock -n 9 || die "another experiment scheduler is already running"

mkdir -p "$artifact_root/batches"
batch_dir="$artifact_root/batches/$batch_id"
[ ! -e "$batch_dir" ] || die "batch directory already exists: $batch_dir"
mkdir -p "$batch_dir"
status_file="$batch_dir/status.tsv"
git_sha="$(git -C "$repo_root" rev-parse --short=8 HEAD)" || die "cannot resolve Git HEAD"
git_label="$git_sha"
if [ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]; then
  git_label="${git_sha}-dirty"
fi

declare -a input_configs=() input_seeds=() seed_order=()
declare -A input_seen=() seed_seen=()
for requested in "${requested_jobs[@]}"; do
  config="${requested%:*}"
  seed="${requested##*:}"
  [ "$config" != "$requested" ] || die "job must use CONFIG:SEED: $requested"
  config="$(resolve_experiment_config "$config")" || exit $?
  [[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be an integer in [0, 4294967295]: $requested"
  [ "$seed" -le 4294967295 ] 2>/dev/null \
    || die "seed must be an integer in [0, 4294967295]: $requested"
  while [ "${#seed}" -gt 1 ] && [ "${seed#0}" != "$seed" ]; do
    seed="${seed#0}"
  done
  key="$config:$seed"
  [ -z "${input_seen[$key]:-}" ] || die "duplicate job: $key"
  input_seen[$key]=1
  input_configs+=("$config")
  input_seeds+=("$seed")
  if [ -z "${seed_seen[$seed]:-}" ]; then
    seed_seen[$seed]=1
    seed_order+=("$seed")
  fi
done

declare -a job_ids=() job_stages=() job_names=() job_seeds=() job_repeats=()
declare -a job_depends=() job_gpus=() job_pids=() job_statuses=()
declare -a job_exit_codes=() job_queued_at=() job_started_at=() job_finished_at=()
declare -a job_configs=() job_run_dirs=() job_failure_reasons=()
declare -A job_index=() gpu_job=() seed_m_star_path=() seed_baseline_path=()

add_job() {
  local job_id="$1" stage="$2" name="$3" seed="$4" repeat="$5"
  local depends="$6" config="$7" run_dir="$8"
  [ -z "${job_index[$job_id]:-}" ] || die "duplicate scheduler job id: $job_id"
  local index="${#job_ids[@]}"
  job_index[$job_id]="$index"
  job_ids+=("$job_id")
  job_stages+=("$stage")
  job_names+=("$name")
  job_seeds+=("$seed")
  job_repeats+=("$repeat")
  job_depends+=("$depends")
  job_gpus+=("")
  job_pids+=("")
  job_statuses+=("queued")
  job_exit_codes+=("")
  job_queued_at+=("$(timestamp)")
  job_started_at+=("")
  job_finished_at+=("")
  job_configs+=("$config")
  job_run_dirs+=("$run_dir")
  job_failure_reasons+=("")
}

experiment_path() {
  local config="$1" relative
  relative="${config#"$experiments_root"/}"
  printf '%s' "${relative%.*}"
}

for seed in "${seed_order[@]}"; do
  m_star_id="mstar-seed-$seed"
  m_star_run_id="${batch_id}_seed-${seed}_git-${git_label}_${m_star_id}"
  m_star_run_dir="$artifact_root/canonical_clean/m_star/$m_star_run_id"
  seed_m_star_path[$seed]="$m_star_run_dir/checkpoints/m_star.pt"
  seed_baseline_path[$seed]="$batch_dir/canonical_clean_seed-${seed}.json"
  add_job "$m_star_id" "mstar" "canonical-clean" "$seed" "" "" \
    "$canonical_config" "$m_star_run_dir"

  clean_dependencies=""
  for repeat in $(seq 1 "$clean_replicas"); do
    clean_id="clean-seed-${seed}-repeat-${repeat}"
    clean_run_id="${batch_id}_seed-${seed}_git-${git_label}_${clean_id}"
    clean_run_dir="$artifact_root/canonical_clean/clean-repeat-${repeat}/$clean_run_id"
    add_job "$clean_id" "clean" "canonical-clean" "$seed" "$repeat" \
      "$m_star_id" "$canonical_config" "$clean_run_dir"
    if [ -z "$clean_dependencies" ]; then
      clean_dependencies="$clean_id"
    else
      clean_dependencies="$clean_dependencies,$clean_id"
    fi
  done
  aggregate_id="canonical-aggregate-seed-$seed"
  add_job "$aggregate_id" "canonical_aggregate" "canonical-clean" "$seed" "" \
    "$clean_dependencies" "$canonical_config" "$batch_dir"
done

for input_index in "${!input_configs[@]}"; do
  config="${input_configs[$input_index]}"
  seed="${input_seeds[$input_index]}"
  ordinal="$(printf '%03d' "$((input_index + 1))")"
  experiment_id="${experiment_branch}-${ordinal}-seed-${seed}"
  experiment_run_id="${batch_id}_seed-${seed}_git-${git_label}_${experiment_id}"
  experiment_run_dir="$artifact_root/$(experiment_path "$config")/$experiment_run_id"
  add_job "$experiment_id" "$experiment_branch" "$(basename "${config%.*}")" "$seed" "" \
    "canonical-aggregate-seed-$seed" "$config" "$experiment_run_dir"
done

write_status() {
  local temporary="$status_file.tmp" index
  if ! {
    printf 'job_id\tstage\texperiment\tseed\trepeat\tdepends_on\tgpu\tpid\tstatus\texit_code\tqueued_at\tstarted_at\tfinished_at\tconfig\trun_dir\tfailure_reason\n'
    for index in "${!job_ids[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${job_ids[$index]}" "${job_stages[$index]}" "${job_names[$index]}" \
        "${job_seeds[$index]}" "${job_repeats[$index]}" "${job_depends[$index]}" \
        "${job_gpus[$index]}" "${job_pids[$index]}" "${job_statuses[$index]}" \
        "${job_exit_codes[$index]}" "${job_queued_at[$index]}" \
        "${job_started_at[$index]}" "${job_finished_at[$index]}" \
        "${job_configs[$index]}" "${job_run_dirs[$index]}" \
        "${job_failure_reasons[$index]}"
    done
  } > "$temporary"; then
    rm -f -- "$temporary"
    return 1
  fi
  if ! mv -- "$temporary" "$status_file"; then
    rm -f -- "$temporary"
    return 1
  fi
}

dependency_state() {
  local index="$1" depends dep dep_index dep_status waiting=0
  local -a dependency_ids=()
  depends="${job_depends[$index]}"
  if [ -z "$depends" ]; then
    echo ready
    return
  fi
  IFS=',' read -r -a dependency_ids <<< "$depends"
  for dep in "${dependency_ids[@]}"; do
    dep_index="${job_index[$dep]}"
    dep_status="${job_statuses[$dep_index]}"
    if [ "$dep_status" = "failed" ]; then
      echo "failed:$dep"
      return
    fi
    if [ "$dep_status" != "completed" ]; then
      waiting=1
    fi
  done
  if [ "$waiting" -eq 0 ]; then
    echo ready
  else
    echo waiting
  fi
}

gpu_is_idle() {
  local gpu="$1" compute_pids memory
  [ -z "${gpu_job[$gpu]:-}" ] || return 1
  compute_pids="$($nvidia_smi_bin --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" \
    || return 1
  if echo "$compute_pids" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
    return 1
  fi
  memory="$($nvidia_smi_bin --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1)" \
    || return 1
  memory="${memory//[[:space:]]/}"
  [[ "$memory" =~ ^[0-9]+$ ]] || return 1
  [ "$memory" -le "$idle_memory_mib" ]
}

pid_is_running() {
  local pid="$1" state
  state="$(ps -o stat= -p "$pid" 2>/dev/null)" || return 1
  state="${state//[[:space:]]/}"
  case "$state" in
    R*|S*|D*|T*|I*) return 0 ;;
    *) return 1 ;;
  esac
}

process_group_is_running() {
  ps -o stat= --pgroup "$1" 2>/dev/null \
    | grep -Eq '^[[:space:]]*[RSDTI]'
}

terminate_running_jobs() {
  local index pid attempt any_running
  for index in "${!job_ids[@]}"; do
    [ "${job_statuses[$index]}" = "running" ] || continue
    pid="${job_pids[$index]}"
    [ -n "$pid" ] || continue
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for attempt in $(seq 1 50); do
    any_running=0
    for index in "${!job_ids[@]}"; do
      [ "${job_statuses[$index]}" = "running" ] || continue
      pid="${job_pids[$index]}"
      if [ -n "$pid" ] && process_group_is_running "$pid"; then
        any_running=1
        break
      fi
    done
    [ "$any_running" -eq 1 ] || break
    sleep 0.1
  done
  for index in "${!job_ids[@]}"; do
    [ "${job_statuses[$index]}" = "running" ] || continue
    pid="${job_pids[$index]}"
    [ -n "$pid" ] || continue
    if process_group_is_running "$pid"; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
}

abort_scheduler() {
  local reason="$1" exit_code="$2" write_final_status="$3" index gpu
  if [ "${scheduler_aborting:-0}" -eq 1 ]; then
    exit "$exit_code"
  fi
  scheduler_aborting=1
  trap '' HUP INT TERM
  terminate_running_jobs
  for index in "${!job_ids[@]}"; do
    case "${job_statuses[$index]}" in
      queued|running)
        gpu="${job_gpus[$index]}"
        [ -z "$gpu" ] || unset "gpu_job[$gpu]"
        job_statuses[$index]="failed"
        job_exit_codes[$index]="$exit_code"
        job_finished_at[$index]="$(timestamp)"
        job_failure_reasons[$index]="$reason"
        ;;
    esac
  done
  if [ "$write_final_status" -eq 1 ] && ! write_status; then
    echo "error: could not persist final interrupted status: $status_file" >&2
  fi
  echo "batch aborted: $reason; status: $status_file" >&2
  exit "$exit_code"
}

record_signal() {
  local signal="$1"
  [ -n "${pending_signal:-}" ] || pending_signal="$signal"
}

handle_pending_signal() {
  case "${pending_signal:-}" in
    HUP) abort_scheduler "scheduler_interrupted:HUP" 129 1 ;;
    INT) abort_scheduler "scheduler_interrupted:INT" 130 1 ;;
    TERM) abort_scheduler "scheduler_interrupted:TERM" 143 1 ;;
  esac
}

require_status_write() {
  if ! write_status; then
    echo "error: cannot persist scheduler status: $status_file" >&2
    abort_scheduler "status_write_failed" 74 0
  fi
}

manifest_is_completed() {
  "$python_bin" -c \
    'import json, sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if isinstance(payload, dict) and payload.get("status") == "completed" else 1)' \
    "$1" >/dev/null 2>&1
}

start_gpu_job() {
  local index="$1" gpu="$2" stage seed run_dir log_path pid
  local -a command=()
  stage="${job_stages[$index]}"
  seed="${job_seeds[$index]}"
  run_dir="${job_run_dirs[$index]}"
  if [ -e "$run_dir" ] || ! mkdir -p "$run_dir"; then
    job_statuses[$index]="failed"
    job_exit_codes[$index]="73"
    job_finished_at[$index]="$(timestamp)"
    job_failure_reasons[$index]="run_directory_unavailable"
    echo "${job_ids[$index]}: failed (run directory unavailable: $run_dir)" >&2
    return 1
  fi
  command=(
    "$python_bin" -m mflpoison.runner
    --config "${job_configs[$index]}"
    --run-dir "$run_dir"
    --seed "$seed"
  )
  case "$stage" in
    mstar)
      command+=(--m-star-only)
      ;;
    clean)
      command+=(--branch clean --m-star-path "${seed_m_star_path[$seed]}")
      ;;
    attack|defended)
      command+=(
        --branch "$stage"
        --m-star-path "${seed_m_star_path[$seed]}"
        --canonical-clean "${seed_baseline_path[$seed]}"
      )
      ;;
    *)
      die "cannot start non-GPU stage on GPU: $stage"
      ;;
  esac
  log_path="$run_dir/train.log"
  handle_pending_signal
  CUDA_VISIBLE_DEVICES="$gpu" setsid "${command[@]}" >"$log_path" 2>&1 &
  pid=$!
  job_gpus[$index]="$gpu"
  job_pids[$index]="$pid"
  job_statuses[$index]="running"
  job_started_at[$index]="$(timestamp)"
  gpu_job[$gpu]="$index"
  handle_pending_signal
  echo "started ${job_ids[$index]} on GPU $gpu: pid=$pid, run_dir=$run_dir"
}

run_canonical_aggregate() {
  local index="$1" seed depends dep dep_index output log_path pid rc
  local -a dependency_ids=() summaries=() aggregate_command=()
  seed="${job_seeds[$index]}"
  output="${seed_baseline_path[$seed]}"
  log_path="$batch_dir/canonical_clean_seed-${seed}.log"
  job_statuses[$index]="running"
  job_started_at[$index]="$(timestamp)"
  depends="${job_depends[$index]}"
  IFS=',' read -r -a dependency_ids <<< "$depends"
  for dep in "${dependency_ids[@]}"; do
    dep_index="${job_index[$dep]}"
    summaries+=("${job_run_dirs[$dep_index]}/summary.json")
  done
  aggregate_command=(
    "$python_bin" -m mflpoison.runner.canonical_clean
    --seed "$seed"
    --m-star-path "${seed_m_star_path[$seed]}"
    --output "$output"
    "${summaries[@]}"
  )
  handle_pending_signal
  setsid "${aggregate_command[@]}" >"$log_path" 2>&1 &
  pid=$!
  job_pids[$index]="$pid"
  handle_pending_signal
  require_status_write
  handle_pending_signal
  wait "$pid"
  rc=$?
  handle_pending_signal
  if [ "$rc" -eq 0 ]; then
    if [ -f "$output" ]; then
      job_statuses[$index]="completed"
    else
      rc=1
      job_statuses[$index]="failed"
      job_failure_reasons[$index]="missing_canonical_clean"
    fi
  else
    job_statuses[$index]="failed"
    job_failure_reasons[$index]="exit_code:$rc"
  fi
  job_exit_codes[$index]="$rc"
  job_finished_at[$index]="$(timestamp)"
  echo "${job_ids[$index]}: ${job_statuses[$index]}"
}

reap_finished_jobs() {
  local index pid gpu rc
  scheduler_progress=0
  for index in "${!job_ids[@]}"; do
    [ "${job_statuses[$index]}" = "running" ] || continue
    [ "${job_stages[$index]}" != "canonical_aggregate" ] || continue
    pid="${job_pids[$index]}"
    if pid_is_running "$pid"; then
      continue
    fi
    wait "$pid"
    rc=$?
    handle_pending_signal
    gpu="${job_gpus[$index]}"
    unset "gpu_job[$gpu]"
    job_exit_codes[$index]="$rc"
    job_finished_at[$index]="$(timestamp)"
    if [ "$rc" -eq 0 ] && manifest_is_completed "${job_run_dirs[$index]}/run_manifest.json"; then
      job_statuses[$index]="completed"
    else
      job_statuses[$index]="failed"
      if [ "$rc" -eq 0 ]; then
        job_failure_reasons[$index]="invalid_or_incomplete_run_manifest"
        job_exit_codes[$index]="1"
      else
        job_failure_reasons[$index]="exit_code:$rc"
      fi
    fi
    scheduler_progress=1
    echo "${job_ids[$index]}: ${job_statuses[$index]} (${job_run_dirs[$index]})"
  done
}

mark_dependency_failures() {
  local changed=1 index state dependency
  while [ "$changed" -eq 1 ]; do
    changed=0
    for index in "${!job_ids[@]}"; do
      [ "${job_statuses[$index]}" = "queued" ] || continue
      state="$(dependency_state "$index")"
      case "$state" in
        failed:*)
          dependency="${state#failed:}"
          job_statuses[$index]="failed"
          job_exit_codes[$index]="125"
          job_finished_at[$index]="$(timestamp)"
          job_failure_reasons[$index]="dependency_failed:$dependency"
          changed=1
          scheduler_progress=1
          ;;
      esac
    done
  done
}

all_jobs_finished() {
  local status
  for status in "${job_statuses[@]}"; do
    case "$status" in
      completed|failed) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

scheduler_aborting=0
pending_signal=""
trap 'record_signal HUP' HUP
trap 'record_signal INT' INT
trap 'record_signal TERM' TERM
for index in "${!job_ids[@]}"; do
  [ "${job_stages[$index]}" = "canonical_aggregate" ] && continue
  [ ! -e "${job_run_dirs[$index]}" ] \
    || die "run directory already exists: ${job_run_dirs[$index]}"
done

require_status_write
handle_pending_signal
echo "batch $batch_id queued; status: $status_file"

while ! all_jobs_finished; do
  handle_pending_signal
  reap_finished_jobs
  handle_pending_signal
  mark_dependency_failures

  for index in "${!job_ids[@]}"; do
    [ "${job_statuses[$index]}" = "queued" ] || continue
    [ "${job_stages[$index]}" = "canonical_aggregate" ] || continue
    if [ "$(dependency_state "$index")" = "ready" ]; then
      run_canonical_aggregate "$index"
      handle_pending_signal
      scheduler_progress=1
    fi
  done
  mark_dependency_failures

  for gpu in "${gpu_pool[@]}"; do
    gpu_is_idle "$gpu" || continue
    for index in "${!job_ids[@]}"; do
      [ "${job_statuses[$index]}" = "queued" ] || continue
      [ "${job_stages[$index]}" != "canonical_aggregate" ] || continue
      [ "$(dependency_state "$index")" = "ready" ] || continue
      start_gpu_job "$index" "$gpu" || true
      handle_pending_signal
      scheduler_progress=1
      break
    done
  done

  if [ "$scheduler_progress" -eq 1 ]; then
    require_status_write
    handle_pending_signal
  fi
  all_jobs_finished && break
  handle_pending_signal
  sleep "$monitor_interval"
  handle_pending_signal
done

require_status_write
handle_pending_signal
failed=0
for status in "${job_statuses[@]}"; do
  [ "$status" != "failed" ] || failed=$((failed + 1))
done
handle_pending_signal
echo "batch finished: completed=$((${#job_ids[@]} - failed)), failed=$failed; status: $status_file"
[ "$failed" -eq 0 ]
