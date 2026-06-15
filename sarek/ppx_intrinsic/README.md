# Sarek PPX Intrinsic

**Package**: `sarek.ppx_intrinsic`  
**Purpose**: PPX for defining GPU intrinsic functions

Minimal PPX extension for declaring GPU intrinsic functions and types that work across multiple backends.

## Overview

The `%sarek_intrinsic` PPX allows defining functions and types that generate backend-specific code at runtime while providing OCaml implementations for CPU execution.

## Usage

```ocaml
(* Define an intrinsic type *)
let%sarek_intrinsic float32 =
  {device = (fun _ -> "float"); ctype = Ctypes.float}

(* Define an intrinsic function *)
let%sarek_intrinsic (sin_float32 : float -> float) =
  {device = (fun d -> "sin(%s)"); ocaml = sin}
```

## Features

- Dual code generation (GPU device code + OCaml host code)
- Backend-specific code via device functions
- Automatic registration with Sarek_registry
- Type-safe intrinsic declarations

## Architecture

The PPX generates:
1. Runtime registration for JIT compilation
2. PPX-time registration for compile-time type checking
3. OCaml function implementations for CPU execution

This separation breaks circular dependencies between sarek_stdlib and sarek_ppx.
