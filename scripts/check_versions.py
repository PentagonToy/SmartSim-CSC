#!/usr/bin/env python3

from __future__ import annotations

import configparser
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_stack_version() -> str:
    version_file = ROOT / "VERSION"
    version = version_file.read_text().strip()

    if not version:
        raise RuntimeError(f"Empty version file: {version_file}")

    return version


def read_smartsim_version() -> str:
    version_file = ROOT / "components" / "smartsim" / "smartsim" / "version.py"
    match = re.search(
        r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
        version_file.read_text(),
    )

    if match is None:
        raise RuntimeError(f"Could not read SmartSim version from {version_file}")

    return match.group(1)


def read_smartredis_version() -> str:
    setup_cfg = ROOT / "components" / "smartredis" / "setup.cfg"
    config = configparser.ConfigParser()
    config.read(setup_cfg)

    try:
        return config["metadata"]["version"].strip()
    except KeyError as exc:
        raise RuntimeError(
            f"Could not read SmartRedis version from {setup_cfg}"
        ) from exc


def read_redisai_version() -> str:
    version_header = ROOT / "components" / "redisai" / "src" / "version.h"
    text = version_header.read_text()

    values = []

    for name in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(
            rf"#define\s+REDISAI_VERSION_{name}\s+(\d+)",
            text,
        )

        if match is None:
            raise RuntimeError(
                f"Could not read REDISAI_VERSION_{name} from {version_header}"
            )

        values.append(match.group(1))

    return ".".join(values)


def main() -> None:
    versions = {
        "SmartSim-CSC": read_stack_version(),
        "SmartSim": read_smartsim_version(),
        "SmartRedis": read_smartredis_version(),
        "RedisAI": read_redisai_version(),
    }

    for name, version in versions.items():
        print(f"{name:13} {version}")

    print("\nVersion metadata check passed.")


if __name__ == "__main__":
    main()
