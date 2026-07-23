#!/usr/bin/env python3

import hashlib
import json
import os
import socket
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import export


def recv_line(conn: socket.socket) -> bytes:
    chunks = []

    while True:
        chunk = conn.recv(65536)

        if not chunk:
            break

        if b"\n" in chunk:
            before, _, _ = chunk.partition(b"\n")
            chunks.append(before)
            break

        chunks.append(chunk)

    return b"".join(chunks)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py SOCKET_PATH")

    socket_path = sys.argv[1]
    models = {}

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(16)

    print(f"JAX worker ready: {socket_path}", flush=True)
    print(f"JAX version: {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    try:
        while True:
            conn, _ = server.accept()

            with conn:
                try:
                    request = json.loads(recv_line(conn))

                    model_blob = bytes.fromhex(request["model"])
                    model_key = hashlib.sha256(model_blob).hexdigest()

                    model = models.get(model_key)

                    if model is None:
                        model = export.deserialize(model_blob)
                        models[model_key] = model

                    dtype = np.dtype(request["dtype"])
                    shape = tuple(request["shape"])

                    x = np.asarray(request["data"], dtype=dtype).reshape(shape)
                    y = np.asarray(model.call(jnp.asarray(x)))

                    response = {
                        "ok": True,
                        "dtype": y.dtype.name,
                        "shape": list(y.shape),
                        "data": y.reshape(-1).tolist(),
                    }

                except Exception as exc:
                    response = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                conn.sendall(json.dumps(response).encode() + b"\n")

    finally:
        server.close()

        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    main()
