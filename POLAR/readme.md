# POLAR Makefile (Linux / Conda-friendly) — Build Guide

This README explains how to build **POLAR** using the provided Makefile on Linux, with support for ONNX model loading via **protobuf-generated C++ headers** and a small **ONNX bridge header**.

---

## What this Makefile does

- Builds `libpolar.a` and `polar_tool`.
- Generates **only** `onnx-ml` protobuf C++ code (to avoid duplicate symbol issues).
- Uses a local ONNX bridge header: `third_party/onnx_include/onnx/onnx_pb.h`.
- Supports both **conda** and **system** installations via a single `PREFIX` variable.

---

## 0) Prerequisites

You need:
- `g++` (GCC 8+ recommended)
- `make`
- `protoc` (must match the protobuf headers/libs you link against)
- Libraries and headers for: `mpfr`, `gmp`, `gsl`, `glpk`
- Protobuf + Abseil (the Makefile links `-lprotobuf` and `-labsl_*`)

If using conda-forge, you typically want:
- `mpfr gmp gsl glpk`
- `protobuf libprotobuf`
- `libabseil abseil-cpp`
- (optional) `nlohmann_json` if POLAR includes it

---

## 1) Choose a PREFIX

The Makefile uses:
- `PREFIX ?= $(CONDA_PREFIX)`
- if empty, falls back to `/usr/local`

Common options:

### Conda
```bash
conda activate polar_nas
make clean
make PREFIX=$CONDA_PREFIX
```

### System (/usr)
```bash
make clean
make PREFIX=/usr
```

### Custom install prefix
```bash
make clean
make PREFIX=$HOME/local
```

---

## 2) Ensure Flow* is built first

POLAR links against Flow* static library. Build Flow* first (example; keep your own Flow* build procedure if already working):

```bash
cd ../flowstar/flowstar-toolbox
make clean
make
cd ../../POLAR
```

Make sure Flow* produced:
`../flowstar/flowstar-toolbox/libflowstar.a`

---

## 3) Ensure the ONNX bridge header exists (critical)

POLAR source includes:
```cpp
#include <onnx/onnx_pb.h>
```

To make this work without relying on system ONNX headers, the repo must provide:

**File:** `third_party/onnx_include/onnx/onnx_pb.h`

**Recommended content:**
```cpp
#pragma once
#include "onnx/onnx-ml.pb.h"
```

This bridge header ensures the include resolves to the generated protobuf header:
`build/onnx_pb/onnx/onnx-ml.pb.h`

---

## 4) ONNX protobuf generation workflow

The Makefile generates C++ sources from:
`third_party/onnx_proto/onnx-ml.proto`

It creates:
- `build/onnx_pb/onnx/onnx-ml.pb.h`
- `build/onnx_pb/onnx/onnx-ml.pb.cc`
- `build/onnx_pb/onnx/onnx-ml.pb.o`

**Important note:** Only `onnx-ml.proto` is generated to avoid duplicate definitions that occur if both `onnx.proto` and `onnx-ml.proto` are compiled together.

---

## 5) Build POLAR

From `POLAR/`:

```bash
make clean
make PREFIX=$CONDA_PREFIX
```

Expected outputs:
- `libpolar.a`
- `polar_tool`
- `build/onnx_pb/onnx/onnx-ml.pb.o`

---

## 6) Troubleshooting

### A) mpfr.h: No such file or directory
Your compiler is not seeing the correct include path. Ensure:
1. `PREFIX` points to a prefix containing `include/mpfr.h`.
2. You invoked: `make PREFIX=$CONDA_PREFIX`.

Check:
```bash
ls $CONDA_PREFIX/include/mpfr.h
```

### B) Abseil link errors (DSO missing from command line)
Your protobuf build depends on abseil logging libraries. This Makefile includes a minimal `-labsl_*` set. If your platform’s abseil split differs, override `ABSL_LIBS` by editing the Makefile `LIBS` list (or extend it).

### C) protoc mismatch (headers/libs/codegen inconsistent)
Make sure the `protoc` used matches the protobuf headers and `libprotobuf.so` on your `PREFIX`.

Check:
```bash
which protoc
protoc --version
ls $CONDA_PREFIX/lib/libprotobuf.so*
```

---

## 7) Clean build

```bash
make clean
```
This removes:
- `build/`
- object files
- `libpolar.a`, `polar_tool`