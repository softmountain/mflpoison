#!/usr/bin/env python3
import argparse
import sys

from _dispatch import (
    TRAIN_SCRIPTS,
    config_to_cli_arguments,
    dispatch,
    load_experiment_config,
)


def main():
    parser = argparse.ArgumentParser(description="Dispatch a generator trainer")
    parser.add_argument("--generator", choices=sorted(TRAIN_SCRIPTS))
    parser.add_argument("--config", help="JSON or YAML experiment config")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    config = load_experiment_config(args.config)
    configured_variant = config.get("generator", {}).get("variant")
    generator = args.generator or configured_variant
    if generator is None:
        parser.error("one of --generator or --config is required")
    if generator not in TRAIN_SCRIPTS:
        parser.error(f"unsupported generator variant in config: {generator}")

    # Explicit command-line arguments come last and override config values.
    configured_arguments = config_to_cli_arguments(config)
    return dispatch(
        TRAIN_SCRIPTS[generator],
        configured_arguments + forwarded,
    )


if __name__ == "__main__":
    sys.exit(main())
