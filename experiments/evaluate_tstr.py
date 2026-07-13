#!/usr/bin/env python3
import argparse
import sys

from _dispatch import dispatch


def main():
    parser = argparse.ArgumentParser(description="Run the shared TSTR protocol")
    parser.add_argument("--synthetic_data", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return dispatch(
        "fed_multimodal/Local/train_synthetic.py",
        ["--synthetic_data", args.synthetic_data] + forwarded,
    )


if __name__ == "__main__":
    sys.exit(main())
