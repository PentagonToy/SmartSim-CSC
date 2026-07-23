#!/usr/bin/env python3

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "stack.toml"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def stack_name() -> str:
    return load_config()["stack"]["name"]


def stack_version() -> str:
    config = load_config()
    version_file = ROOT / config["stack"]["version_file"]
    version = version_file.read_text().strip()

    if not version:
        raise RuntimeError(f"Empty version file: {version_file}")

    return version


def component_path(name: str) -> Path:
    config = load_config()

    try:
        relative_path = config["components"][name]["path"]
    except KeyError as exc:
        raise KeyError(f"Unknown component: {name}") from exc

    path = ROOT / relative_path

    if not path.is_dir():
        raise FileNotFoundError(f"Component directory does not exist: {path}")

    return path


def profile(name: str) -> dict:
    config = load_config()

    try:
        return config["profiles"][name]
    except KeyError as exc:
        raise KeyError(f"Unknown build profile: {name}") from exc


def shell_config(profile_name: str) -> None:
    selected = profile(profile_name)
    backends = selected["backends"]

    print(f"STACK_NAME={stack_name()}")
    print(f"STACK_VERSION={stack_version()}")
    print(f"SMARTSIM_DIR={component_path('smartsim')}")
    print(f"SMARTREDIS_DIR={component_path('smartredis')}")
    print(f"REDISAI_DIR={component_path('redisai')}")
    print(f"DEVICE={selected['device']}")
    print(f"BACKENDS={','.join(backends)}")


def main(profile_name: str) -> None:
    print(f"Stack: {stack_name()} {stack_version()}")

    for name in load_config()["components"]:
        print(f"{name:11} {component_path(name)}")

    selected = profile(profile_name)
    print(f"Profile:      {profile_name}")
    print(f"Device:       {selected['device']}")
    print(f"Backends:     {', '.join(selected['backends'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--profile", default="linux-x64-cpu")
    args = parser.parse_args()

    if args.shell:
        shell_config(args.profile)
    else:
        main(args.profile)
