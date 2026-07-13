#!/usr/bin/env python3
import argparse
import sys

from _dispatch import EVAL_SCRIPTS, dispatch


def main():
    parser = argparse.ArgumentParser(description="Dispatch a generator evaluator")
    parser.add_argument("--generator", required=True, choices=sorted(EVAL_SCRIPTS))
    parser.add_argument("--checkpoint", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forwarded = ["--checkpoint", args.checkpoint] + forwarded
    return dispatch(EVAL_SCRIPTS[args.generator], forwarded)


if __name__ == "__main__":
    sys.exit(main())
