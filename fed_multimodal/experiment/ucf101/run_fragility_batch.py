"""P9: fragility 实验矩阵并发调度器。

跑 A/B/C 消融组 (各 5 组) + baseline×3 (不同 seed), 并发 --parallel N。
每组调 train_label_flip.py:
  - 攻击组: --malicious_clients 0,1,2,3, --attack_n_inject N, seed=8 (固定, 保证 A/B/C 可比)
  - baseline: --malicious_clients "", seed=8/9/10 (3 次取均值算 drift)

sr=1.0 全员参与, 恶意客户端每轮必被采样。
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


def build_attack_cmd(a, source, target, subdir):
    return [PY, SCRIPT,
            "--m_star_path", a.m_star_path,
            "--synth_data", a.synth_data,
            "--attack_source_class", str(source),
            "--attack_target_class", str(target),
            "--attack_epochs", str(a.attack_epochs),
            "--eval_every", str(a.eval_every),
            "--attack_n_inject", str(a.attack_n_inject),
            "--malicious_clients", a.malicious_clients,
            "--seed", "8",
            "--result_subdir", subdir,
            "--data_dir", a.data_dir,
            "--alpha", str(a.alpha),
            "--fed_alg", a.fed_alg,
            "--batch_size", str(a.batch_size),
            "--sample_rate", str(a.sample_rate)]


def build_baseline_cmd(a, seed, subdir):
    return [PY, SCRIPT,
            "--m_star_path", a.m_star_path,
            "--attack_source_class", "0", "--attack_target_class", "0",
            "--attack_epochs", str(a.attack_epochs),
            "--eval_every", str(a.eval_every),
            "--malicious_clients", "",
            "--seed", str(seed),
            "--result_subdir", subdir,
            "--data_dir", a.data_dir,
            "--alpha", str(a.alpha),
            "--fed_alg", a.fed_alg,
            "--batch_size", str(a.batch_size),
            "--sample_rate", str(a.sample_rate)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", required=True, help="combos_fragility.json from select_targets_v2.py")
    p.add_argument("--m_star_path", required=True)
    p.add_argument("--synth_data", required=True)
    p.add_argument("--attack_epochs", type=int, default=15)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--attack_n_inject", type=int, default=30)
    p.add_argument("--malicious_clients", default="0,1,2,3")
    p.add_argument("--data_dir", default="/home/xp/fedpoi/fed_multimodal/results")
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--fed_alg", default="fed_avg")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--sample_rate", type=float, default=1.0)
    p.add_argument("--baseline_seeds", default="8,9,10")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_attacks", action="store_true")
    p.add_argument("--groups", default="A,B,C")
    p.add_argument("--parallel", type=int, default=5)
    args = p.parse_args()

    combos = json.load(open(args.combos))
    env = {**os.environ, "PYTHONPATH": "/home/xp/fedpoi"}

    pending = []
    if not args.skip_attacks:
        for g in args.groups.split(","):
            for row in combos.get(f"combos_{g}", []):
                subdir = f"frag_{row['group']}_s{row['source']}_t{row['target']}"
                pending.append((subdir, build_attack_cmd(args, row['source'], row['target'], subdir)))
    if not args.skip_baseline:
        for seed in [int(s) for s in args.baseline_seeds.split(",")]:
            subdir = f"frag_baseline_seed{seed}"
            pending.append((subdir, build_baseline_cmd(args, seed, subdir)))

    total = len(pending)
    npar = max(1, args.parallel)
    running = []; done = 0; failures = []
    print(f"Running {total} jobs (parallel={npar}): "
          f"{'attacks=' + str(total - (0 if args.skip_baseline else len(args.baseline_seeds.split(',')))) + ' ' if not args.skip_baseline else ''}"
          f"{'baselines=' + str(len(args.baseline_seeds.split(','))) if not args.skip_baseline and not args.skip_attacks else ''}")
    while pending or running:
        while pending and len(running) < npar:
            subdir, cmd = pending.pop(0)
            logf = open(f"/tmp/frag_{subdir}.log", "w")
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running.append((proc, subdir, logf))
            print(f"  [start {done + len(running)}/{total}] {subdir} (pid {proc.pid})", flush=True)
        time.sleep(15)
        still = []
        for proc, subdir, logf in running:
            if proc.poll() is None:
                still.append((proc, subdir, logf))
            else:
                logf.close(); done += 1
                rc = proc.returncode
                print(f"  [done {done}/{total}] {'OK' if rc == 0 else f'FAIL(rc={rc})'} {subdir}  -> /tmp/frag_{subdir}.log", flush=True)
                if rc != 0:
                    failures.append(subdir)
        running = still
    print(f"\nDone. {total - len(failures)}/{total} succeeded.")
    if failures:
        print("Failures:", failures)


if __name__ == "__main__":
    main()
