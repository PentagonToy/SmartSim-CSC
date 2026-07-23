# SmartSim-CSC

SmartSim-CSC is a CSC-oriented monorepo containing the SmartSim, SmartRedis, and RedisAI sources required for integrated simulation and machine-learning workflows on HPC systems.

Maintained by:

Aalto University  
Department of Energy and Mechanical Engineering  
Energy Conversion and Systems Team

Based on the upstream projects:

- [SmartSim](https://github.com/CrayLabs/SmartSim)
- [SmartRedis](https://github.com/CrayLabs/SmartRedis)
- [RedisAI](https://github.com/RedisAI/RedisAI)

## Versions

<!-- versions:start -->

| Component | Version |
| --- | --- |
| SmartSim-CSC | `1.0.0` |
| SmartSim | `1.0.0+csc` |
| SmartRedis | `1.0.0+csc` |
| RedisAI | `1.2.7` |

<!-- versions:end -->

The SmartSim-CSC stack version is defined in `VERSION`. Component versions are read from their native package or source metadata.

```bash
python scripts/check_versions.py
```

## Repository Structure

```text
SmartSim-CSC/
├── components/
│   ├── smartsim/
│   ├── smartredis/
│   └── redisai/
├── examples/
│   ├── GettingStarted/
│   ├── OnlineAnalysis/
│   └── OnlineInference/
├── scripts/
│   ├── install.sh
│   ├── stack_config.py
│   ├── version_info.py
│   ├── check_versions.py
│   └── audit_hardcoding.py
├── stack.toml
├── VERSION
└── README.md
```

The three components are stored directly in this repository. Git submodules and the former separate CSC component repositories are not required.

## Components

### SmartSim

SmartSim provides the infrastructure used to create and launch simulation experiments, submit and monitor HPC workloads, start and stop the Orchestrator, configure Redis and RedisAI, and manage machine-learning backend execution.

### SmartRedis

SmartRedis provides Python, C, C++, and Fortran clients for tensor exchange, dataset and metadata exchange, model registration, model execution, and communication between simulation and machine-learning applications.

The CSC implementation additionally supports direct registration of supported JAX and Equinox callables.

### RedisAI

RedisAI extends Redis with machine-learning model execution. The bundled CSC source includes a JAX backend that communicates with a persistent Python worker.

## Current Validated Profile

Build profiles are defined in `stack.toml`.

```toml
[profiles.linux-x64-cpu]
device = "cpu"
backends = ["onnxruntime", "jax"]

[profiles.linux-arm64-cpu]
device = "cpu"
backends = ["onnxruntime", "jax"]
```

These profiles have been validated with:

- Linux x86_64
- Linux ARM64 / aarch64
- Python 3.12
- NumPy 2.x
- Redis 7.2.4
- RedisAI 1.2.7 CSC source
- ONNX Runtime backend
- JAX backend
- SmartSim tensor transfer
- SmartSim ONNX backend validation

Roihu x86_64 CPU and ARM64 CPU configurations are validated. GPU execution still requires separate end-to-end validation on an allocated GPU node.

## Installation

Create or select a clean Python 3.12 environment and run:

```bash
PYTHON=/path/to/python \
PROFILE=linux-x64-cpu \
    ./scripts/install.sh
```

Example:

```bash
python3.12 -m venv .venv

PYTHON="$PWD/.venv/bin/python" \
PROFILE=linux-x64-cpu \
    ./scripts/install.sh
```

The installer:

1. reads component paths and backend selection from `stack.toml`
2. installs the bundled SmartRedis and SmartSim sources
3. installs the required JAX runtime when enabled
4. builds Redis
5. builds the bundled RedisAI source
6. builds the selected RedisAI backends
7. checks Python package dependencies
8. verifies required build artifacts
9. runs `smart validate`

A successful installation ends with:

```text
[SmartSim] INFO Success!
SmartSim-CSC 1.0.0 installation completed successfully.
```

## Configuration

The central stack configuration is stored in `stack.toml`.

```toml
[stack]
name = "SmartSim-CSC"
version_file = "VERSION"

[components.smartsim]
path = "components/smartsim"

[components.smartredis]
path = "components/smartredis"

[components.redisai]
path = "components/redisai"

[profiles.linux-x64-cpu]
device = "cpu"
backends = ["onnxruntime", "jax"]
```

Inspect the resolved configuration with:

```bash
python scripts/stack_config.py --profile linux-x64-cpu
```

## Basic SmartSim Usage

```python
from smartsim import Experiment

experiment = Experiment("hello_world", launcher="auto")
settings = experiment.create_run_settings(
    exe="echo",
    exe_args="Hello from SmartSim",
)
model = experiment.create_model("hello_world", settings)
experiment.start(model, block=True, summary=True)
```

## SmartRedis Tensor Exchange

```python
import numpy as np
from smartredis import Client

client = Client(address="127.0.0.1:6379", cluster=False)
data = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
client.put_tensor("input", data)
result = client.get_tensor("input")
```

## JAX and Equinox Execution

A supported JAX or Equinox callable can be registered through SmartRedis:

```python
client.set_model(
    name,
    model,
    backend="JAX",
    example_inputs=example_inputs,
)
```

It can then be executed through the standard model interface:

```python
client.run_model(
    name,
    inputs=["input_tensor"],
    outputs=["output_tensor"],
)
```

Execution path:

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

## Current JAX Backend Scope

Validated functionality on Linux x86_64 CPU includes:

- direct Equinox callable registration
- model serialization through SmartRedis
- RedisAI model execution
- output shape preservation
- `float32`, `float64`, `int32`, and `int64` outputs

Current limitations:

- input tensors are currently expected to use `float32`
- the current protocol supports one model input and one model output
- scalar outputs may require explicit array shaping
- GPU execution has not yet been fully validated
- the current JSON and socket protocol is an initial CSC implementation

Model shape, dtype, and numerical results should be verified before production use.

## Native SmartRedis Library

The native SmartRedis library is required for C, C++, Fortran, and linked simulation applications.

```bash
cd components/smartredis

env \
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran
```

The native library must be built separately for each target architecture.

## HPC Usage

SmartSim supports Slurm, PBS, SGE, and local execution. CSC Roihu uses Slurm.

The same SmartSim-CSC repository can be used from both x86_64 CPU nodes and ARM64 login or GPU nodes. The selected stack profile defines the requested device and backends, while SmartSim detects the host operating system and architecture at runtime and selects the corresponding machine-learning package configuration, such as `LinuxX64CPU.json` or `LinuxARM64CPU.json`.

Separate repository clones are not required. However, architecture-specific Python environments, native libraries, RedisAI backends, and installation directories must remain separate.

| Environment | Architecture |
| --- | --- |
| Roihu CPU nodes | x86_64 |
| Roihu GPU login and compute nodes | ARM64 / aarch64 |

The Linux x86_64 CPU and Linux ARM64 CPU profiles are validated with the ONNX Runtime and JAX backends.

## Validation

```bash
python scripts/check_versions.py
python scripts/stack_config.py --profile linux-x64-cpu
smart validate
```

The current validated profile verifies tensor transfer and the ONNX backend. JAX applications should additionally be tested using an end-to-end model example.

## Build Safety

Do not replace RedisAI shared libraries while Redis or the JAX worker is running.

Use this order when rebuilding backend libraries:

1. stop the SmartSim Orchestrator
2. stop the JAX worker
3. rebuild or replace RedisAI and backend libraries
4. restart the JAX worker
5. restart the Orchestrator

Replacing a loaded shared library may terminate Redis or cause a segmentation fault.

## Development

Check for potentially duplicated or stale configuration:

```bash
python scripts/audit_hardcoding.py
```

Not every literal reported by this script is an error. Backend names in implementation code, protocol constants, CMake options, and upstream API versions are normally expected.

Values that should generally be centralized include stack versions, component paths, build profiles, backend selection, repository and release URLs, and user-facing installation instructions.

## Planned Extensions

Potential future additions include:

- Linux ARM64 CPU profile
- CUDA profiles
- ROCm profiles
- additional Roihu platform validation
- native SmartRedis build automation
- OpenFOAM integration
- expanded JAX input and output support

These should be introduced as independently validated profiles or components.

## Documentation

- [SmartSim documentation](https://www.craylabs.org/docs/overview.html)
- [SmartSim API](https://www.craylabs.org/docs/api/smartsim_api.html)
- [SmartRedis API](https://www.craylabs.org/docs/api/smartredis_api.html)
- [SmartSim tutorials](https://www.craylabs.org/docs/tutorials/getting_started/getting_started.html)

Repository examples are available under `examples/`.

## Citation

Partee et al., “Using Machine Learning at Scale in Numerical Simulations with SmartSim: An Application to Ocean Climate Modeling”, *Journal of Computational Science*, Volume 62, 2022, 101707.

DOI: [10.1016/j.jocs.2022.101707](https://doi.org/10.1016/j.jocs.2022.101707)

```bibtex
@article{PARTEE2022101707,
    title = {Using Machine Learning at scale in numerical simulations with SmartSim: An application to ocean climate modeling},
    journal = {Journal of Computational Science},
    volume = {62},
    pages = {101707},
    year = {2022},
    issn = {1877-7503},
    doi = {10.1016/j.jocs.2022.101707},
    author = {Sam Partee and Matthew Ellis and Alessandro Rigazzi and Andrew E. Shao and Scott Bachman and Gustavo Marques and Benjamin Robbins},
}
```

## License

SmartSim-CSC retains the licenses and copyright terms of its upstream components. See the license files distributed with each component for the applicable terms.
