import argparse
import json
import re
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


def _experiment_name(config_path: Path) -> str:
    name = config_path.stem.lower()
    return re.sub(r"[^a-z0-9_-]+", "_", name).strip("_") or "experiment"


def _default_run_dir(config: ScenarioConfig, config_path: Path) -> Path:
    now = datetime.now()
    return (
        Path(config.results.root_dir)
        / now.strftime("%Y-%m-%d")
        / _experiment_name(config_path)
        / f"{now:%H-%M-%S}_seed-{config.federation.seed}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the UCF101 federated poisoning and defense scenario"
    )
    parser.add_argument("--config", required=True, help="Scenario YAML or JSON path")
    parser.add_argument("--run-dir", help="Exact result directory for this run")
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
        results_dir=run_dir,
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
