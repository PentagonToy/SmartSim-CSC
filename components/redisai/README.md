# RedisAI for CSC

This repository contains the CSC-maintained RedisAI fork used by the CSC SmartSim and SmartRedis stack.

It is intended for high-performance computing environments, with primary support for CSC Roihu.

Maintained by:

Aalto University
Department of Energy and Mechanical Engineering
Energy Conversion and Systems Team

Based on:

* RedisAI: https://github.com/RedisAI/RedisAI
* SmartSim: https://github.com/CrayLabs/SmartSim
* SmartRedis: https://github.com/CrayLabs/SmartRedis

CSC repositories:

* RedisAI: https://github.com/PentagonToy/RedisAI
* SmartRedis: https://github.com/PentagonToy/SmartRedis
* SmartSim: https://github.com/PentagonToy/SmartSim
* CSC installation guide: https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

Current CSC release:

```text
RedisAI v1.0.0-csc
```

---

## Overview

RedisAI is a Redis module for executing machine-learning models close to the data stored in Redis.

It provides model execution through backend libraries while allowing simulation applications and machine-learning workflows to exchange tensors through the same database.

Within the CSC SmartSim stack, RedisAI is used by the SmartSim Orchestrator and accessed through SmartRedis.

The CSC fork additionally provides a JAX backend for direct execution of JAX and Equinox models.

---

## CSC Support

The CSC RedisAI release includes:

* Linux x86_64 support
* Linux ARM64 / aarch64 support
* TensorFlow backend integration
* ONNX Runtime backend integration
* LibTorch backend integration
* JAX backend integration
* persistent Python worker communication for JAX execution
* JAX output shape preservation
* JAX output support for:

  * `float32`
  * `float64`
  * `int32`
  * `int64`
* SmartSim-managed JAX worker lifecycle
* integration with the CSC SmartRedis model-registration interface

RedisAI and its backend libraries must be built separately for each target architecture.

| Environment     | Architecture    |
| --------------- | --------------- |
| Roihu CPU nodes | x86_64          |
| Roihu GPU nodes | ARM64 / aarch64 |

---

## Supported Backends

The CSC RedisAI build supports:

* TensorFlow
* ONNX Runtime
* PyTorch through LibTorch
* JAX
* Equinox through the JAX backend

TensorFlow, ONNX Runtime, and LibTorch use compiled RedisAI backend libraries.

JAX and Equinox models are executed through a persistent Python worker connected to the RedisAI JAX backend.

---

## JAX Backend

The JAX backend allows a Python callable to be registered through SmartRedis and executed through RedisAI.

Example:

```python
client.set_model(
    name,
    model,
    backend="JAX",
    example_inputs=example_inputs,
)
```

The model can then be executed using the standard SmartRedis interface:

```python
client.run_model(
    name,
    inputs=["input_tensor"],
    outputs=["output_tensor"],
)
```

SmartRedis handles model serialization and registration, while SmartSim manages the JAX worker together with the Orchestrator lifecycle.

This allows JAX and Equinox inference to use the same tensor-transfer and model-execution workflow as the other RedisAI backends.

---

## Architecture

The JAX execution path consists of:

```text
Simulation or Python application
        |
        v
SmartRedis client
        |
        v
RedisAI JAX backend
        |
        v
Persistent Python JAX worker
        |
        v
JAX or Equinox callable
```

Model inputs are stored as RedisAI tensors, sent to the JAX worker for execution, and returned as RedisAI output tensors.

---

## Building with SmartSim

The recommended way to build the CSC RedisAI stack is through the matching SmartSim release.

Recommended versions:

```text
RedisAI:    v1.0.0-csc
SmartRedis: v1.0.0-csc
SmartSim:   v1.0.3-csc
```

Install SmartRedis and SmartSim:

```bash
uv pip install \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@v1.0.0-csc"

uv pip install \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@v1.0.3-csc"
```

Build Redis, RedisAI, and the configured backend libraries:

```bash
export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages
```

The resulting RedisAI module and backend libraries are installed inside the SmartSim package directory.

The JAX build additionally includes:

```text
redisai_jax.so
worker.py
```

Detailed CSC installation instructions are available here:

https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

---

## Building from Source

Clone the CSC release:

```bash
git clone \
    --branch v1.0.0-csc \
    --recursive \
    https://github.com/PentagonToy/RedisAI.git

cd RedisAI
```

RedisAI backend dependencies and build options are normally managed by SmartSim.

For direct RedisAI development, consult the source configuration and backend build files in this repository.

---

## Loading RedisAI

RedisAI is loaded into Redis as a module.

Example:

```bash
redis-server \
    --loadmodule /path/to/redisai.so
```

Backend libraries are loaded according to the RedisAI configuration generated by SmartSim.

In the CSC stack, users normally start RedisAI through the SmartSim Orchestrator rather than invoking `redis-server` manually.

---

## Related Components

### SmartSim

SmartSim manages:

* Redis and RedisAI installation
* Orchestrator startup and shutdown
* Slurm-launched workflows
* backend library configuration
* JAX worker startup and shutdown

Repository:

https://github.com/PentagonToy/SmartSim

### SmartRedis

SmartRedis provides:

* tensor exchange
* dataset exchange
* model registration
* model execution
* Python, C, C++, and Fortran clients
* direct JAX and Equinox callable registration

Repository:

https://github.com/PentagonToy/SmartRedis

---

## Documentation

CSC-specific documentation:

* CSC SmartSim environment guide:
  https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

* CSC SmartSim repository:
  https://github.com/PentagonToy/SmartSim

* CSC SmartRedis repository:
  https://github.com/PentagonToy/SmartRedis

Upstream resources:

* RedisAI repository:
  https://github.com/RedisAI/RedisAI

* SmartSim documentation:
  https://www.craylabs.org/docs/overview.html

* SmartRedis API documentation:
  https://www.craylabs.org/docs/api/smartredis_api.html

---

## License

This repository retains the license and copyright terms of the upstream RedisAI project.

See:

https://github.com/PentagonToy/RedisAI/blob/1.0.0-csc/LICENSE
