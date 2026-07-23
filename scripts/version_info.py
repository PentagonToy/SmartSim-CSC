#!/usr/bin/env python3

from __future__ import annotations

import configparser
import re
from pathlib import Path

from stack_config import ROOT, component_path, load_config


def stack_version() -> str:
    config = load_config()
    version_file = ROOT / config["stack"]["version_file"]
    version = version_file.read_text().strip()

    if not version:
        raise RuntimeError(f"Empty version file: {version_file}")

    return version


def smartsim_version() -> str:
    buildenv_file = (
        component_path("smartsim")
        / "smartsim"
        / "_core"
        / "_install"
        / "buildenv.py"
    )
    match = re.search(
        r'SMARTSIM\s*=\s*Version_\(get_env\('
        r'"SMARTSIM_VERSION",\s*"([^"]+)"\)\)',
        buildenv_file.read_text(),
    )

    if match is None:
        raise RuntimeError(
            f"Could not read SmartSim version from {buildenv_file}"
        )

    return match.group(1)


def smartredis_version() -> str:
    setup_cfg = component_path("smartredis") / "setup.cfg"
    config = configparser.ConfigParser()
    config.read(setup_cfg)

    try:
        return config["metadata"]["version"].strip()
    except KeyError as exc:
        raise RuntimeError(
            f"Could not read SmartRedis version from {setup_cfg}"
        ) from exc


def redisai_version() -> str:
    version_header = component_path("redisai") / "src" / "version.h"
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


def all_versions() -> dict[str, str]:
    return {
        "SmartSim-CSC": stack_version(),
        "SmartSim": smartsim_version(),
        "SmartRedis": smartredis_version(),
        "RedisAI": redisai_version(),
    }


if __name__ == "__main__":
    for name, version in all_versions().items():
        print(f"{name:13} {version}")
