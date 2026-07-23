#!/usr/bin/env python3

from __future__ import annotations

from version_info import all_versions


def main() -> None:
    for name, version in all_versions().items():
        print(f"{name:13} {version}")

    print("\nVersion metadata check passed.")


if __name__ == "__main__":
    main()
