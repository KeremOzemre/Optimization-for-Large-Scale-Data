"""Compatibility entry point; the dataset preparation code lives in prepare.py."""

from run import main
import sys


if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
