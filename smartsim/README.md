# SmartSim for CSC Roihu

This repository contains a CSC-oriented SmartSim configuration for high-performance computing environments, with primary support for CSC Roihu.

This configuration is maintained by:

Aalto University
Department of Energy and Mechanical Engineering
Energy Conversion and Systems Team

It is based on:

* SmartSim: https://github.com/CrayLabs/SmartSim
* SmartRedis: https://github.com/CrayLabs/SmartRedis
* RedisAI: https://github.com/RedisAI/RedisAI

The purpose of this fork is to provide a reproducible SmartSim stack for CSC systems, including Roihu CPU and GPU node architectures, with integrated JAX and Equinox model execution through RedisAI.

Current CSC releases:

| Component  | Release      |
| ---------- | ------------ |
| SmartSim   | `v1.0.1-csc` |
| SmartRedis | `v1.0.0-csc` |
| RedisAI    | `v1.0.0-csc` |

Detailed installation instructions are available here:

https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

For general SmartSim usage, API documentation, tutorials, and examples, refer to the official SmartSim documentation:

https://www.craylabs.org/docs/overview.html

---

## Overview

SmartSim consists of two main components:

1. SmartSim Infrastructure Library
2. SmartRedis Client Library

SmartSim manages simulation and machine-learning workflows on HPC systems.

SmartRedis connects applications written in Python, C, C++, and Fortran to the SmartSim Orchestrator database. Applications can exchange tensors and datasets, execute machine-learning models, and process data while simulations are running.

This client-server design allows simulation and machine-learning components to exchange data without requiring MPI communication for every transfer.

The CSC fork additionally supports direct registration and execution of JAX and Equinox models through RedisAI.

---

## CSC Roihu Support

This repository is intended primarily for CSC Roihu.

Roihu provides two relevant node architectures:

| Environment     | Architecture    |
| --------------- | --------------- |
| Roihu CPU nodes | x86_64          |
| Roihu GPU nodes | ARM64 / aarch64 |

SmartSim, SmartRedis, RedisAI, their compiled libraries, and machine-learning backends must be built separately for each architecture.

The CSC installation guide covers:

* Python 3.12 environment creation
* SmartSim and SmartRedis installation
* Redis and RedisAI builds
* JAX and Equinox support
* PyTorch, TensorFlow, and ONNX Runtime backends
* x86_64 and ARM64 environments
* CSC module configuration
* Tykky packaging
* Roihu CPU and GPU usage
* native SmartRedis C++ and Fortran libraries
* Jupyter kernel registration

See:

https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

---

## CSC-Specific Features

The CSC releases include:

* Python 3.12 support
* NumPy 2.x compatibility
* Linux ARM64 platform support
* x86_64 and ARM64 build configurations
* RedisAI TensorFlow backend support on Linux ARM64
* RedisAI ONNX Runtime backend support on Linux ARM64
* RedisAI LibTorch backend support on Linux ARM64
* RedisAI JAX backend integration
* direct JAX and Equinox model registration through SmartRedis
* persistent JAX worker lifecycle management through SmartSim
* JAX output shape preservation
* JAX output support for:

  * `float32`
  * `float64`
  * `int32`
  * `int64`
* SmartRedis compiler and source compatibility fixes
* native C++ and Fortran SmartRedis library builds

No post-install source patching is required when using the CSC releases.

---

## SmartSim Components

### SmartSim

SmartSim provides the Python infrastructure used to:

* create experiments
* launch simulation applications
* submit Slurm jobs
* launch ensembles
* start and stop the Orchestrator database
* manage the RedisAI JAX worker
* monitor running applications
* connect simulation and machine-learning workflows

### SmartRedis

SmartRedis provides client libraries for:

* Python
* C
* C++
* Fortran

The clients can send and retrieve:

* tensors
* datasets
* metadata
* machine-learning models
* executable scripts and functions

The CSC SmartRedis release also supports direct registration of Python JAX and Equinox callables.

### RedisAI

RedisAI extends Redis with machine-learning model execution.

The CSC RedisAI release adds a JAX backend that communicates with a persistent Python worker. Serialized JAX or Equinox models can therefore be registered through SmartRedis and executed inside the SmartSim Orchestrator workflow.

### Orchestrator

The SmartSim Orchestrator is an in-memory database based on Redis and RedisAI.

It can be used for:

* runtime data exchange
* online analysis
* online data processing
* machine-learning inference
* JAX and Equinox inference
* coupling multiple simulation components

---

## HPC Usage

SmartSim supports common HPC launch systems, including:

* Slurm
* PBS
* SGE
* local execution

CSC Roihu uses Slurm.

A basic SmartSim experiment can use automatic launcher detection:

```python
from smartsim import Experiment

exp = Experiment("hello_world", launcher="auto")

settings = exp.create_run_settings(
    exe="echo",
    exe_args="Hello from SmartSim",
)

model = exp.create_model("hello_world", settings)

exp.start(model, block=True, summary=True)
```

Run it inside an appropriate Roihu allocation:

```bash
python hello_world.py
```

For complete allocation, installation, and environment-loading instructions, use the CSC guide linked above.

---

## Machine-Learning Backends

The CSC SmartSim stack provides RedisAI backends for:

* PyTorch through LibTorch
* TensorFlow
* ONNX Runtime
* JAX and Equinox

These backends allow SmartRedis clients to execute machine-learning models directly through the Orchestrator.

JAX and Equinox callables can be registered directly from Python:

```python
client.set_model(
    name,
    model,
    backend="JAX",
    example_inputs=example_inputs,
)
```

The registered model can then be executed with the standard SmartRedis model-running interface.

The SmartSim CSC release manages the persistent JAX worker as part of the Orchestrator lifecycle.

---

## Current JAX Backend Scope

The current JAX backend has been validated on x86_64 CPU nodes.

Verified functionality includes:

* direct Equinox callable registration
* model serialization and transfer through SmartRedis
* RedisAI model execution
* output shape preservation
* output dtypes:

  * `float32`
  * `float64`
  * `int32`
  * `int64`

Current limitations:

* input tensors are currently expected to use `float32`
* the current protocol supports one model input and one model output
* scalar outputs may require explicit array shaping
* GPU execution has not yet been fully validated
* the current JSON and socket protocol should be considered an initial CSC implementation

Users should validate their model shape, dtype, and numerical output before production use.

---

## Installation

The recommended releases are:

```text
RedisAI:    v1.0.0-csc
SmartRedis: v1.0.0-csc
SmartSim:   v1.0.1-csc
```

The SmartSim `v1.0.1-csc` patch release updates its Python dependency metadata to require the matching CSC SmartRedis release.

Install SmartRedis and SmartSim from their CSC tags:

```bash
uv pip install \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@v1.0.0-csc"

uv pip install \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@v1.0.1-csc"
```

The full Redis and RedisAI stack can then be built through SmartSim:

```bash
export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages
```

Detailed CSC-specific installation steps are maintained here:

https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

---

## Build Safety

Do not overwrite or replace RedisAI shared libraries while Redis or the JAX worker is running.

Use the following order when rebuilding or replacing backend libraries:

1. stop Redis and the SmartSim Orchestrator
2. stop the JAX worker
3. rebuild or copy the RedisAI libraries
4. start the JAX worker
5. start Redis and the Orchestrator

Replacing a loaded shared library can cause Redis to terminate with a segmentation fault.

---

## Validation

After installation, verify the installed package versions:

```bash
python - <<'PY'
import smartredis
import smartsim

print(f"SmartRedis: {smartredis.__version__}")
print(f"SmartSim:   {smartsim.__version__}")
PY
```

Expected CSC versions:

```text
SmartRedis: 1.0.0+csc
SmartSim:   1.0.3+csc
```

Validate the SmartSim installation:

```bash
smart validate --device cpu
```

The validation should identify the installed RedisAI backends, including the JAX backend.

A separate end-to-end model test is recommended before running production simulations.

---

## Official Documentation

Official SmartSim resources:

* SmartSim repository:
  https://github.com/CrayLabs/SmartSim

* SmartRedis repository:
  https://github.com/CrayLabs/SmartRedis

* SmartSim documentation:
  https://www.craylabs.org/docs/overview.html

* Installation documentation:
  https://www.craylabs.org/docs/installation_instructions/basic.html

* API documentation:
  https://www.craylabs.org/docs/api/smartsim_api.html

* Tutorials:
  https://www.craylabs.org/docs/tutorials/getting_started/getting_started.html

CSC-specific repositories:

* SmartSim CSC fork:
  https://github.com/PentagonToy/SmartSim

* SmartRedis CSC fork:
  https://github.com/PentagonToy/SmartRedis

* RedisAI CSC fork:
  https://github.com/PentagonToy/RedisAI

* CSC installation guide:
  https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

---

## Publications

The following are public presentations or publications using SmartSim:

* Collaboration with NCAR — CGD Seminar
* SmartSim: Using Machine Learning in HPC Simulations
* SmartSim: Online Analytics and Machine Learning for HPC Simulations
* PyTorch Ecosystem Day Poster

---

## Citation

Please use the following citation when referencing SmartSim, SmartRedis, or related SmartSim work:

Partee et al., “Using Machine Learning at Scale in HPC Simulations with SmartSim: An Application to Ocean Climate Modeling”, *Journal of Computational Science*, Volume 62, 2022, 101707, ISSN 1877-7503.

https://doi.org/10.1016/j.jocs.2022.101707

### BibTeX

```bibtex
@article{PARTEE2022101707,
    title = {Using Machine Learning at scale in numerical simulations with SmartSim: An application to ocean climate modeling},
    journal = {Journal of Computational Science},
    volume = {62},
    pages = {101707},
    year = {2022},
    issn = {1877-7503},
    doi = {10.1016/j.jocs.2022.101707},
    url = {https://www.sciencedirect.com/science/article/pii/S1877750322001065},
    author = {Sam Partee and Matthew Ellis and Alessandro Rigazzi and Andrew E. Shao and Scott Bachman and Gustavo Marques and Benjamin Robbins},
    keywords = {Deep learning, Numerical simulation, Climate modeling, High performance computing, SmartSim},
}
```
