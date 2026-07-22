import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from mflpoison.core.config import load_scenario_config

from .scenario import build_default_runner


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the UCF101 federated poisoning and defense scenario"
    )
    parser.add_argument("--config", required=True, help="Scenario YAML or JSON path")
    parser.add_argument(
        "--artifact-root",
        help="Override artifacts.root_dir without changing the scenario config",
    )
    args = parser.parse_args(argv)
    config = load_scenario_config(args.config)
    runner = build_default_runner(
        config,
        artifact_root=None if args.artifact_root is None else Path(args.artifact_root),
    )
    result = runner.run()
    print(
        json.dumps(
            {
                "m_star_hash": result.m_star.content_hash,
                "summary_path": str(result.summary_path),
                "artifact_root": str(result.artifact_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
