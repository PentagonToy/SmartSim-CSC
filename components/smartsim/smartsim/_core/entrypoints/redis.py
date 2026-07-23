# BSD 2-Clause License
#
# Copyright (c) 2021-2025 Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from subprocess import PIPE, STDOUT
from types import FrameType

import psutil

from smartsim._core.utils.network import current_ip
from smartsim.entity.dbnode import LaunchedShardData
from smartsim.log import get_logger

logger = get_logger(__name__)

"""
Redis/KeyDB entrypoint script
"""

DBPID: int | None = None
JAX_WORKER_PID: int | None = None
JAX_SOCKET_PATH: str | None = None

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]


def handle_signal(signo: int, _frame: FrameType | None) -> None:
    if not signo:
        logger.warning("Received signal with no signo")
    cleanup()
    raise SystemExit(128 + signo)


def build_bind_args(source_addr: str, *addrs: str) -> tuple[str, ...]:
    return (
        "--bind",
        source_addr,
        *addrs,
        # pin source address to avoid random selection by Redis
        "--bind-source-addr",
        source_addr,
    )


def build_cluster_args(shard_data: LaunchedShardData) -> tuple[str, ...]:
    if cluster_conf_file := shard_data.cluster_conf_file:
        return ("--cluster-enabled", "yes", "--cluster-config-file", cluster_conf_file)
    return ()


def print_summary(
    cmd: list[str], network_interface: str, shard_data: LaunchedShardData
) -> None:
    print(
        textwrap.dedent(f"""\
            ----------- Running Command ----------
            COMMAND: {' '.join(cmd)}
            IPADDRESS: {shard_data.hostname}
            NETWORK: {network_interface}
            SMARTSIM_ORC_SHARD_INFO: {json.dumps(shard_data.to_dict())}
            --------------------------------------

            --------------- Output ---------------

            """),
        flush=True,
    )


def main(args: argparse.Namespace) -> int:
    global DBPID, JAX_WORKER_PID, JAX_SOCKET_PATH  # pylint: disable=global-statement

    src_addr, *bind_addrs = (current_ip(net_if) for net_if in args.ifname.split(","))
    shard_data = LaunchedShardData(
        name=args.name, hostname=src_addr, port=args.port, cluster=args.cluster
    )

    cmd = [
        args.orc_exe,
        args.conf_file,
        *args.rai_module,
        "--port",
        str(args.port),
        *build_cluster_args(shard_data),
        *build_bind_args(src_addr, *bind_addrs),
    ]

    print_summary(cmd, args.ifname, shard_data)

    env = os.environ.copy()
    worker_path = (
        Path(__file__).resolve().parents[1]
        / "lib"
        / "backends"
        / "redisai_jax"
        / "worker.py"
    )

    try:
        if worker_path.is_file():
            JAX_SOCKET_PATH = f"/tmp/smartsim-jax-{os.getuid()}-{args.port}.sock"

            if os.path.exists(JAX_SOCKET_PATH):
                os.unlink(JAX_SOCKET_PATH)

            env["SMARTSIM_JAX_SOCKET"] = JAX_SOCKET_PATH

            worker = psutil.Popen(
                [sys.executable, str(worker_path), JAX_SOCKET_PATH],
                env=env,
            )
            JAX_WORKER_PID = worker.pid

            for _ in range(100):
                if os.path.exists(JAX_SOCKET_PATH):
                    break
                if worker.poll() is not None:
                    raise RuntimeError("JAX worker exited during startup")
                time.sleep(0.1)
            else:
                raise RuntimeError("Timed out waiting for JAX worker socket")

        process = psutil.Popen(cmd, stdout=PIPE, stderr=STDOUT, env=env)
        DBPID = process.pid

        for line in iter(process.stdout.readline, b""):
            print(line.decode("utf-8").rstrip(), flush=True)

        return_code = process.wait()

        if return_code != 0:
            logger.error(
                f"Database process exited with return code {return_code}"
            )
            return return_code

        return 0

    except Exception:
        logger.error("Database process starter raised an exception", exc_info=True)
        return 1

    finally:
        cleanup()


def cleanup() -> None:
    logger.debug("Cleaning up database instance")

    for process_name, pid in (
        ("database", DBPID),
        ("JAX worker", JAX_WORKER_PID),
    ):
        if pid is None:
            continue

        try:
            process = psutil.Process(pid)
            process.terminate()

            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                logger.warning(
                    f"{process_name} did not terminate gracefully; killing it"
                )
                process.kill()
                process.wait(timeout=5)

        except psutil.NoSuchProcess:
            logger.debug(f"{process_name} process already stopped")
        except OSError as error:
            logger.warning(
                f"Failed to stop {process_name} process gracefully: {error}"
            )

    if JAX_SOCKET_PATH and os.path.exists(JAX_SOCKET_PATH):
        try:
            os.unlink(JAX_SOCKET_PATH)
        except OSError as error:
            logger.warning(f"Failed to remove JAX worker socket: {error}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Process Launcher"
    )
    parser.add_argument(
        "+orc-exe", type=str, help="Path to the orchestrator executable", required=True
    )
    parser.add_argument(
        "+conf-file",
        type=str,
        help="Path to the orchestrator configuration file",
        required=True,
    )
    parser.add_argument(
        "+rai-module",
        nargs="+",
        type=str,
        help=(
            "Command for the orcestrator to load the Redis AI module with "
            "symbols seperated by whitespace"
        ),
        required=True,
    )
    parser.add_argument(
        "+name", type=str, help="Name to identify the shard", required=True
    )
    parser.add_argument(
        "+port",
        type=int,
        help="The port on which to launch the shard of the orchestrator",
        required=True,
    )
    parser.add_argument(
        "+ifname", type=str, help="Network Interface name", required=True
    )
    parser.add_argument(
        "+cluster",
        action="store_true",
        help="Specify if this orchestrator shard is part of a cluster",
    )

    args_ = parser.parse_args()

    # make sure to register the cleanup before the start
    # the process so our signaller will be able to stop
    # the database process.
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)

    raise SystemExit(main(args_))
