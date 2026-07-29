import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml

from mflpoison.core.config import ScenarioConfig, load_scenario_config

from .builder import build_default_runner


def _with_seed(config: ScenarioConfig, seed: Optional[int]) -> ScenarioConfig:
    if seed is None:
        return config
    payload = config.to_dict()
    payload["federation"]["seed"] = int(seed)
    payload["generator"]["seed"] = int(seed)
    return ScenarioConfig.from_mapping(payload)


def _slug(name: str) -> str:
    name = name.lower()
    return re.sub(r"[^a-z0-9_-]+", "_", name).strip("_") or "experiment"


def _experiment_path(config_path: Path) -> Path:
    experiment_name = _slug(config_path.stem)
    group_name = _slug(config_path.parent.name)
    if group_name == "experiments":
        return Path(experiment_name)
    return Path(group_name) / experiment_name


def _git_short_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"],
        text=True,
    ).strip()


def _default_run_dir(config: ScenarioConfig, config_path: Path) -> Path:
    now = datetime.now()
    run_id = (
        f"{now:%Y%m%d-%H%M%S}"
        f"_seed-{config.federation.seed}"
        f"_git-{_git_short_sha()}"
    )
    return (
        Path(config.artifact.root_dir)
        / _experiment_path(config_path)
        / run_id
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the UCF101 federated poisoning and defense scenario"
    )
    parser.add_argument("--config", required=True, help="Scenario YAML or JSON path")
    parser.add_argument("--run-dir", help="Exact artifact directory for this run")
    parser.add_argument("--seed", type=int, help="Override federation and generator seed")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = _with_seed(load_scenario_config(config_path), args.seed)
    run_dir = (
        Path(args.run_dir)
        if args.run_dir is not None
        else _default_run_dir(config, config_path)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.to_dict(),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    runner = build_default_runner(
        config,
        artifact_dir=run_dir,
    )
    result = runner.run()
    print(
        json.dumps(
            {
                "m_star_hash": result.m_star.content_hash,
                "summary_path": str(result.summary_path),
                "run_dir": str(result.run_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
