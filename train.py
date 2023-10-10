#! /usr/bin/env python3

from pathlib import Path

from tokenizer import Tokenizer


if __name__ == "__main__":
    t = Tokenizer(Path("canterbury_tales.txt"))
