# Benchmark Makefile (Linux / Conda-friendly) — Build Guide

This README explains how to build the **benchmark executables** using the updated Makefile on Linux, designed to work with the **POLAR** build (including ONNX protobuf support).

---

## What this Makefile does

- Builds benchmark executables (e.g. `reachnn_benchmark_1`, `flowstar_1step`, `flowstar_1step_v1`).
- Links against:
  - `libpolar.a` (from `../../POLAR`)
  - `libflowstar.a` (from `../../flowstar/flowstar-toolbox`)
  - protobuf + abseil (same toolchain as POLAR)
- Adds include paths so `../../POLAR/NeuralNetwork.h` can find:
  - `onnx/onnx_pb.h` (bridge header inside the POLAR repo)
  - generated protobuf headers (`../../POLAR/build/onnx_pb/...`)

---

## 0) Prerequisites

Before building benchmarks, you must have:

1. **Flow*** built:
   - `../../flowstar/flowstar-toolbox/libflowstar.a`

2. **POLAR** built:
   - `../../POLAR/libpolar.a`
   - `../../POLAR/build/onnx_pb/onnx/onnx-ml.pb.o`

The benchmarks link ONNX protobuf symbols by directly linking the generated object:
- `../../POLAR/build/onnx_pb/onnx/onnx-ml.pb.o`

This is required because `libpolar.a` does **not** contain the protobuf object.

---

## 1) Choose a PREFIX

The Makefile uses:
- `PREFIX ?= $(CONDA_PREFIX)`
- if empty, falls back to `/usr/local`

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

---

## 2) Build order (required)

### Step 1 — build Flow*
```bash
cd ../../flowstar/flowstar-toolbox
make
```

### Step 2 — build POLAR
```bash
cd ../../POLAR
make clean
make PREFIX=$CONDA_PREFIX
```
Confirm the ONNX pb object exists:
```bash
ls -l build/onnx_pb/onnx/onnx-ml.pb.o
```

### Step 3 — build benchmarks
```bash
cd ../benchmarks/benchmark1
make clean
make PREFIX=$CONDA_PREFIX
```

---

## 3) ONNX bridge header requirement

POLAR headers included by the benchmarks use:
```cpp
#include <onnx/onnx_pb.h>
```

Therefore, the benchmark Makefile must include POLAR’s bridge header path:
`-I$(POLAR_HOME)/third_party/onnx_include`

And the generated protobuf header path:
`-I$(POLAR_HOME)/build/onnx_pb`

If you move directories, ensure these two include directories still exist.

---

## 4) Notes on linking (why onnx-ml.pb.o is needed)

You may see undefined references such as:
- `onnx::ModelProto::ModelProto(google::protobuf::Arena*)`
- `onnx::_GraphProto_default_instance_`

These symbols live in the generated protobuf object:
`$(POLAR_HOME)/build/onnx_pb/onnx/onnx-ml.pb.o`

So the benchmark Makefile explicitly links it as `ONNX_PB_OBJ`.

---

## 5) Troubleshooting

### A) onnx/onnx_pb.h: No such file or directory
You are missing the bridge header include path.
**Fix:**
- Ensure `CFLAGS` contains `-I$(POLAR_HOME)/third_party/onnx_include`.
- Ensure the file exists:
  ```bash
  ls -l ../../POLAR/third_party/onnx_include/onnx/onnx_pb.h
  ```

### B) ONNX protobuf symbol undefined references
This means `onnx-ml.pb.o` was not linked or not built.
**Fix:**
- Build POLAR first.
- Ensure the object exists:
  ```bash
  ls -l ../../POLAR/build/onnx_pb/onnx/onnx-ml.pb.o
  ```

### C) Abseil/protobuf link issues
If you see abseil-related link errors, your environment may require additional `-labsl_*` libraries. The Makefile includes a minimal set. If needed, extend it (or override the variable if you structured it that way).

### D) PREFIX mismatch
Make sure the same PREFIX is used consistently for POLAR and benchmarks (especially for protobuf/abseil):
```bash
make PREFIX=$CONDA_PREFIX
```

---

## 6) Clean build

```bash
make clean
```
Removes:
- local object files
- benchmark executables