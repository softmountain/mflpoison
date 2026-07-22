"""P4 批量调度: 串行跑方式1/方式2 的 (source, target) 攻击组合。

每个组合调 train_label_flip.py 一次。模型参数(hid/att/att_name)由 M* 自动覆盖, 无需传。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/home/xp/anaconda3/envs/poigan/bin/python"
SCRIPT = str(Path(__file__).with_name("train_label_flip.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combos_json", required=True, help="combos_method1.json or combos_method2.json")
    parser.add_argument("--method", type=int, required=True, help="1 or 2 (used in result_subdir naming)")
    parser.add_argument("--m_star_path", required=True)
    parser.add_argument("--synth_data", required=True)
    parser.add_argument("--attack_epochs", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--data_dir", default="/home/xp/fedpoi/fed_multimodal/results")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fed_alg", default="fed_avg")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--start_index", type=int, default=0, help="skip first N combos (resume)")
    parser.add_argument("--parallel", type=int, default=1, help="number of concurrent attack processes")
    args = parser.parse_args()

    combos = json.load(open(args.combos_json))
    print(f"Loaded {len(combos)} combos from {args.combos_json} (method {args.method})")

    env = {**os.environ, "PYTHONPATH": "/home/xp/fedpoi"}
    npar = max(1, args.parallel)

    # build command list (respect start_index)
    pending = []
    for i, c in enumerate(combos):
        if i < args.start_index:
            continue
        s, t = c["source"], c["target"]
        subdir = f"source{s}_target{t}_m{args.method}"
        cmd = [
            PY, SCRIPT,
            "--m_star_path", args.m_star_path,
            "--synth_data", args.synth_data,
            "--attack_source_class", str(s),
            "--attack_target_class", str(t),
            "--attack_epochs", str(args.attack_epochs),
            "--eval_every", str(args.eval_every),
            "--result_subdir", subdir,
            "--data_dir", args.data_dir,
            "--alpha", str(args.alpha),
            "--fed_alg", args.fed_alg,
            "--batch_size", str(args.batch_size),
            "--sample_rate", str(args.sample_rate),
        ]
        pending.append((subdir, cmd))

    total = len(pending)
    running = []
    failures = []
    done = 0
    print(f"Running {total} combos with parallel={npar}")
    while pending or running:
        # launch up to npar concurrent
        while pending and len(running) < npar:
            subdir, cmd = pending.pop(0)
            logf = open(f"/tmp/poison_{subdir}.log", "w")
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running.append((p, subdir, logf))
            print(f"  [start {done + len(running)}/{total}] {subdir} (pid {p.pid})", flush=True)
        time.sleep(10)
        still = []
        for p, subdir, logf in running:
            if p.poll() is None:
                still.append((p, subdir, logf))
            else:
                logf.close()
                done += 1
                if p.returncode == 0:
                    print(f"  [done {done}/{total}] OK    {subdir}", flush=True)
                else:
                    print(f"  [done {done}/{total}] FAIL  {subdir} (rc={p.returncode}, see /tmp/poison_{subdir}.log)", flush=True)
                    failures.append(subdir)
        running = still

    print(f"\nDone. {total - len(failures)}/{total} succeeded.")
    if failures:
        print("Failures:", failures)


if __name__ == "__main__":
    main()
