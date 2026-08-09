import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml

from mflpoison.core.config import ScenarioConfig, load_scenario_config
from mflpoison.artifacts import write_manifest

from .builder import build_default_runner


def _with_runtime_overrides(
    config: ScenarioConfig,
    *,
    seed: Optional[int],
    branches: Optional[Sequence[str]],
    m_star_path: Optional[str],
    m_star_only: bool,
    canonical_clean_path: Optional[str],
) -> ScenarioConfig:
    payload = config.to_dict()
    if seed is not None:
        payload["federation"]["seed"] = int(seed)
        payload["generator"]["seed"] = int(seed)
    if branches is not None:
        payload["federation"]["branches"] = list(branches)
    if m_star_path is not None:
        payload["federation"]["m_star_path"] = str(m_star_path)
    if m_star_only:
        payload["federation"]["m_star_only"] = True
        payload["federation"]["m_star_path"] = None
        payload["federation"]["branches"] = []
        payload["evaluation"]["canonical_clean_path"] = None
    if canonical_clean_path is not None:
        payload["evaluation"]["canonical_clean_path"] = str(
            canonical_clean_path
        )
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
    parser.add_argument(
        "--branch",
        action="append",
        choices=("clean", "attack", "defended"),
        dest="branches",
        help="Run only the selected branch; repeat for multiple branches",
    )
    parser.add_argument(
        "--m-star-path",
        help="Reuse one common M* checkpoint and skip clean pretraining",
    )
    parser.add_argument(
        "--m-star-only",
        action="store_true",
        help="Generate and persist M* without running a branch",
    )
    parser.add_argument(
        "--canonical-clean",
        dest="canonical_clean_path",
        help="Canonical clean JSON used for attack-only Delta_ASR",
    )
    args = parser.parse_args(argv)
    if args.m_star_only and (
        args.branches or args.m_star_path or args.canonical_clean_path
    ):
        parser.error(
            "--m-star-only cannot be combined with --branch, "
            "--m-star-path, or --canonical-clean"
        )
    config_path = Path(args.config)
    config = _with_runtime_overrides(
        load_scenario_config(config_path),
        seed=args.seed,
        branches=args.branches,
        m_star_path=args.m_star_path,
        m_star_only=args.m_star_only,
        canonical_clean_path=args.canonical_clean_path,
    )
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
    try:
        result = runner.run()
    except Exception as exc:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                if manifest.get("status") == "running":
                    manifest["status"] = "failed"
                    manifest["failure"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    write_manifest(manifest, manifest_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        raise
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
