#!/usr/bin/env bash
set -euo pipefail

conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.train "$@"
