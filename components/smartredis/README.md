# SmartRedis for CSC

This repository contains the CSC-maintained SmartRedis fork used with the CSC SmartSim and RedisAI stack.

It is maintained for HPC environments, with primary support for CSC Roihu.

Maintained by:

Aalto University
Department of Energy and Mechanical Engineering
Energy Conversion and Systems Team

Based on:

* SmartRedis: https://github.com/CrayLabs/SmartRedis
* SmartSim: https://github.com/CrayLabs/SmartSim
* RedisAI: https://github.com/RedisAI/RedisAI

CSC repositories:

* SmartRedis: https://github.com/PentagonToy/SmartRedis
* SmartSim: https://github.com/PentagonToy/SmartSim
* RedisAI: https://github.com/PentagonToy/RedisAI
* CSC installation guide: https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

Current CSC release:

```text
SmartRedis v1.0.0-csc
```

---

## Overview

SmartRedis is a collection of Redis clients designed for high-performance computing applications.

It connects simulation and machine-learning applications to Redis and RedisAI, allowing them to exchange data and execute models while simulations are running.

SmartRedis can be used together with SmartSim or as a standalone RedisAI client library.

Supported client languages:

| Language | Version or standard                                                                 |
| -------- | ----------------------------------------------------------------------------------- |
| Python   | 3.10, 3.11, 3.12                                                                    |
| C++      | C++17                                                                               |
| C        | C99                                                                                 |
| Fortran  | Fortran 2018 with GNU or Intel compilers; Fortran 2003 with PGI or NVIDIA compilers |

---

## CSC Support

The CSC fork includes changes required for the CSC SmartSim stack:

* Python 3.12 support
* NumPy 2.x compatibility
* C++17 build support
* Linux x86_64 support
* Linux ARM64 / aarch64 support
* SmartRedis compiler and source compatibility fixes
* native C, C++, and Fortran library builds
* direct JAX and Equinox model registration
* integration with the CSC RedisAI JAX backend

The native SmartRedis library must be built separately for each target architecture.

| Environment     | Architecture    |
| --------------- | --------------- |
| Roihu CPU nodes | x86_64          |
| Roihu GPU nodes | ARM64 / aarch64 |

---

## SmartRedis Clients

SmartRedis clients can send and retrieve:

* tensors
* datasets
* metadata
* machine-learning models
* executable scripts and functions
* model inputs and predictions

SmartRedis supports applications written in:

* Python
* C
* C++
* Fortran

This allows simulation codes and Python machine-learning workflows to communicate through a common Redis database without requiring MPI for every data transfer.

---

## Machine-Learning Backends

SmartRedis can register and execute models through RedisAI backends for:

* PyTorch through LibTorch
* TensorFlow
* ONNX Runtime
* JAX
* Equinox

The CSC SmartRedis release supports direct registration of JAX and Equinox callables from Python.

Example:

```python
client.set_model(
    name,
    model,
    backend="JAX",
    example_inputs=example_inputs,
)
```

The model can then be executed using the standard SmartRedis model interface.

```python
client.run_model(
    name,
    inputs=["input_tensor"],
    outputs=["output_tensor"],
)
```

This removes the need for a separate model-conversion wrapper when using supported JAX or Equinox callables.

---

## Using SmartRedis

SmartRedis can be used through SmartSim:

```python
from smartsim import Experiment
from smartredis import Client
```

It can also connect directly to an existing Redis or RedisAI database:

```python
from smartredis import Client

client = Client(address="127.0.0.1:6379", cluster=False)
```

Basic tensor exchange:

```python
import numpy as np
from smartredis import Client

client = Client(address="127.0.0.1:6379", cluster=False)

data = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

client.put_tensor("input", data)

result = client.get_tensor("input")
```

---

## Installation

Install the CSC SmartRedis release directly from GitHub:

```bash
uv pip install \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@v1.0.0-csc"
```

The matching CSC SmartSim release is:

```bash
uv pip install \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@v1.0.3-csc"
```

The full CSC environment, including RedisAI backends and native SmartRedis libraries, is documented here:

https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

---

## Native Library

The native SmartRedis library is used by C, C++, Fortran, and linked simulation applications.

Example build:

```bash
git clone \
    --branch v1.0.0-csc \
    https://github.com/PentagonToy/SmartRedis.git

cd SmartRedis

env \
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran
```

The resulting libraries are installed under:

```text
install/lib
```

or:

```text
install/lib64
```

depending on the platform.

---

## Dependencies

SmartRedis uses:

* NumPy: https://github.com/numpy/numpy
* Hiredis: https://github.com/redis/hiredis
* Redis-plus-plus: https://github.com/sewenew/redis-plus-plus
* pybind11: https://github.com/pybind/pybind11

JAX and Equinox model execution additionally relies on the matching CSC RedisAI JAX backend.

---

## Documentation

Official SmartRedis documentation:

* SmartRedis API:
  https://www.craylabs.org/docs/api/smartredis_api.html

* SmartSim installation:
  https://www.craylabs.org/docs/installation_instructions/basic.html

* SmartSim documentation:
  https://www.craylabs.org/docs/overview.html

CSC-specific documentation:

* CSC SmartSim environment guide:
  https://github.com/PentagonToy/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md

* CSC SmartSim repository:
  https://github.com/PentagonToy/SmartSim

* CSC RedisAI repository:
  https://github.com/PentagonToy/RedisAI

---

## Publications

Public presentations and publications using SmartRedis include:

* Collaboration with NCAR — CGD Seminar
  https://www.youtube.com/watch?v=2e-5j427AS0

* Using Machine Learning in HPC Simulations
  https://www.sciencedirect.com/science/article/pii/S1877750322001065

* Relexi — A scalable open-source reinforcement-learning framework for high-performance computing
  https://www.sciencedirect.com/science/article/pii/S2665963822001063

---

## Citation

Please use the following citation when referencing SmartSim, SmartRedis, or related SmartSim work:

Partee et al., “Using Machine Learning at Scale in Numerical Simulations with SmartSim: An Application to Ocean Climate Modeling”, *Journal of Computational Science*, Volume 62, 2022, 101707.

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
