# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.

MAKEFLAGS += --no-print-directory
SHELL := /bin/bash

CWD := $(shell pwd)
NPROC := $(shell nproc 2>/dev/null || python -c "import multiprocessing as mp; print(mp.cpu_count())" 2>/dev/null || echo 4)

INSTALL_PREFIX ?= $(CWD)/install
BUILD_FORTRAN ?= OFF
BUILD_PYTHON ?= OFF
BUILD_TYPE ?= Release
LINK_TYPE ?= shared
PEDANTIC ?= OFF

ifeq ($(LINK_TYPE),shared)
	BUILD_SHARED_LIBS := ON
else
	BUILD_SHARED_LIBS := OFF
endif

.PHONY: help
help:
	@echo "SmartRedis CSC build targets"
	@echo
	@echo "  make lib"
	@echo "      Build and install SmartRedis"
	@echo
	@echo "  make lib-with-fortran"
	@echo "      Build and install SmartRedis with Fortran support"
	@echo
	@echo "Optional variables:"
	@echo "  BUILD_TYPE=Release|Debug"
	@echo "  LINK_TYPE=shared|static"
	@echo "  BUILD_PYTHON=ON|OFF"
	@echo "  INSTALL_PREFIX=/path/to/install"
	@echo "  NPROC=<number of build processes>"

.PHONY: lib
lib:
	@cmake -S . -B build/$(BUILD_TYPE) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS) \
		-DBUILD_FORTRAN=$(BUILD_FORTRAN) \
		-DBUILD_PYTHON=$(BUILD_PYTHON) \
		-DPEDANTIC=$(PEDANTIC) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX)
	@cmake --build build/$(BUILD_TYPE) --parallel $(NPROC)
	@cmake --install build/$(BUILD_TYPE)

.PHONY: lib-with-fortran
lib-with-fortran:
	@$(MAKE) lib BUILD_FORTRAN=ON

.PHONY: clean
clean:
	@rm -rf build

.PHONY: clobber
clobber: clean
	@rm -rf install third-party
