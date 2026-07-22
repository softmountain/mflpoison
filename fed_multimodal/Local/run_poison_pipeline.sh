#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

SCENARIO_CONFIG="${SCENARIO_CONFIG:-configs/scenarios/ucf101_generative_poison_defense.yaml}"

args=(--config "$SCENARIO_CONFIG")
if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
  args+=(--artifact-root "$ARTIFACT_ROOT")
fi

python -m mflpoison.runner "${args[@]}"
