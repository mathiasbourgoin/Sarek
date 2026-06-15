# Sarek Float64 Standard Library

**Package**: `sarek.Sarek_float64`  
**Purpose**: Double precision floating point support

Provides `float64` (double precision) type and math functions for Sarek GPU kernels.

## Overview

Float64 module extends Sarek with double precision floating point operations. Each intrinsic works on both GPU (via device code generation) and CPU (via OCaml implementation).

## Usage

```ocaml
open Sarek_float64

let%kernel my_kernel (a : float vector) =
  let x = a.(0) in
  let y = sin_float64 x in
  let z = add_float64 y 1.0 in
  a.(0) <- z
```

## Supported Operations

- Arithmetic: add, sub, mul, div, neg, abs
- Math: sin, cos, tan, exp, log, sqrt, pow
- Comparisons: eq, ne, lt, le, gt, ge

## Backend Support

- CUDA: Full double precision support
- OpenCL: Requires `cl_khr_fp64` extension
- Metal: Limited/no double precision on most devices
- Native: Full support via OCaml float

## Implementation

Uses `%sarek_intrinsic` PPX to generate backend-specific code and register functions with Sarek_registry.
