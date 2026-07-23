#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".venv-test",
    "__pycache__",
    "build",
    "dist",
}

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".md",
    ".rst",
    ".cmake",
    ".txt",
    ".h",
    ".c",
    ".cpp",
}

PATTERNS = {
    "PentagonToy repository URL": re.compile(
        r"https://github\.com/PentagonToy/[A-Za-z0-9_.-]+"
    ),
    "GitHub release URL": re.compile(
        r"https://github\.com/[^/\s]+/[^/\s]+/releases/download/[^\s\"']+"
    ),
    "CSC absolute path": re.compile(
        r"/scratch/project_[^/\s\"']+/[^\s\"']+"
    ),
    "Component version literal": re.compile(
        r"\b(?:0\.1\.0|1\.0\.3\+csc|1\.0\.0\+csc|1\.2\.7)\b"
    ),
    "Backend selection literal": re.compile(
        r"\b(?:onnxruntime|libtorch|libtensorflow|jax|jaxlib)\b"
    ),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "VERSION",
            "Makefile",
        }:
            continue

        yield path


def main() -> None:
    findings = 0

    for path in iter_files():
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        for line_number, line in enumerate(lines, start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    relative = path.relative_to(ROOT)
                    print(f"{label}")
                    print(f"  {relative}:{line_number}")
                    print(f"  {line.strip()}")
                    findings += 1

    print()
    print(f"Total findings: {findings}")


if __name__ == "__main__":
    main()
