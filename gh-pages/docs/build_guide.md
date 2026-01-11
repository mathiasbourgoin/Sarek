---
layout: page
title: Building SPOC from Source
---

# Building and Testing SPOC

While most users should install Sarek via Opam, you can also build the framework from source for development or custom configurations.

## Prerequisites

- **OCaml** >= 5.4.0
- **Dune** >= 3.0
- **ocamlfind**
- **ctypes** & **ctypes-foreign**
- **ppxlib**
- **alcotest** (for running tests)

## Compilation & Installation

### 1. Install Dependencies
```bash
opam install ctypes ctypes-foreign ppxlib alcotest dune
```

### 2. Build and Install
```bash
dune build
dune install
```

## Running Tests

### Unit Tests
Execute the full test suite using dune:
```bash
dune test
```

### Benchmarks
Run the automated benchmark suite to verify performance across available backends:
```bash
# Fast version (small problem sizes)
make benchmarks-fast

# Full version (comprehensive datasets)
make benchmarks
```

## API Documentation
You can generate the latest API documentation locally:
```bash
dune build @doc
```
The documentation will be available in `_build/default/_doc/_html/`.
