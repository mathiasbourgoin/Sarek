# Intake Brief — backend-variant-dedup

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

`gen_variant_def` and `mangle_name` are duplicated across four backend IR modules
(`sarek-cuda/Sarek_ir_cuda.ml`, `sarek-opencl/Sarek_ir_opencl.ml`,
`sarek-metal/Sarek_ir_metal.ml`, `sarek-vulkan/Sarek_ir_glsl.ml`).

- `mangle_name` — **identical** in all four (one-liner: replace `'.'` with `'_'`).
- `gen_variant_def` in CUDA/OpenCL/Metal — targets C/MSL; differs only in (a) the
  `type_of_elttype` function called and (b) the constructor inline prefix
  (`__device__ __host__ inline` for CUDA, `static inline` for OpenCL/Metal).
- `gen_variant_def` in Vulkan — targets GLSL; **structurally different**: uses
  `const int` instead of `enum`, `struct` without `typedef`, flat fields instead of
  C unions, no constructor prefix, and `r.cname_v` instead of `r.data.cname_v`.

Extract into `Sarek_ir_codegen` in `spoc/ir/` with three exports:
- `mangle_name` — shared by all four
- `gen_variant_def` — C/MSL variant (CUDA, OpenCL, Metal), parameterised on
  `~type_of_elttype` and `~constructor_prefix`
- `gen_variant_def_glsl` — GLSL variant (Vulkan), parameterised on `~type_of_elttype`
  only (no prefix)

**Value:** eliminates ~320 lines of duplicated code across four backends; future
variant-codegen fixes land in one place.

## Scope Boundary

Out of scope:
- `generate_with_types` — Metal/Vulkan/OpenCL/CUDA versions all have backend-specific
  structure. Not safely parameterisable. Deferred.
- `type_of_elttype` functions (`cuda_type_of_elttype`, `glsl_type_of_elttype`, etc.)
  — backend-specific, intentionally NOT shared.
- Any other function in the four backend IR files.
- Adding new tests for variant codegen (tracked as a follow-up gap in the KB).
- Changes to test files unless a dune module list needs updating.

**Known issue (Metal):** Metal's `gen_variant_def` is defined at line 1005 but
`generate_with_types` never calls it — Metal kernels with variant types silently
produce no variant typedef. This is a pre-existing bug, not introduced by this
refactor. The implementer must confirm whether the dead definition should be deleted
(acknowledging the gap) or whether a missing `List.iter` call should be added to
`generate_with_types`. The decision belongs to the implementer based on reading the
Metal test suite; it does not block extracting the shared module.

## Relevant Files

| File | Role | Key snippet |
|---|---|---|
| `spoc/ir/dune` | Library definition for `sarek_ir` | `(modules Sarek_ir_types Sarek_ir_pp Sarek_ir_analysis)` — add `Sarek_ir_codegen` |
| `spoc/ir/Sarek_ir_types.ml` | Defines `elttype`, `kernel`, `kern_variants` | `kern_variants : (string * (string * elttype list) list) list` |
| `sarek-cuda/Sarek_ir_cuda.ml:33` | Local `mangle_name` | `let mangle_name name = String.map …` |
| `sarek-cuda/Sarek_ir_cuda.ml:796` | Local C `gen_variant_def` | uses `cuda_type_of_elttype`, prefix `__device__ __host__ inline` |
| `sarek-opencl/Sarek_ir_opencl.ml:44` | Local `mangle_name` | identical to CUDA |
| `sarek-opencl/Sarek_ir_opencl.ml:778` | Local C `gen_variant_def` | uses `opencl_type_of_elttype`, prefix `static inline` |
| `sarek-metal/Sarek_ir_metal.ml:33` | Local `mangle_name` | identical to CUDA |
| `sarek-metal/Sarek_ir_metal.ml:1005` | Local C `gen_variant_def` (dead — no call site in `generate_with_types`) | uses `metal_type_of_elttype`, prefix `static inline` |
| `sarek-vulkan/Sarek_ir_glsl.ml:33` | Local `mangle_name` | identical to CUDA |
| `sarek-vulkan/Sarek_ir_glsl.ml:969` | Local GLSL `gen_variant_def` | uses `glsl_type_of_elttype`, no prefix, flat fields (no union) |
| `sarek-vulkan/Sarek_ir_glsl.ml:1055` | Call site | `List.iter (gen_variant_def buf) k.kern_variants` |

## Architecture Notes

`sarek_ir` (`spoc/ir/`) is a leaf library with no project-internal dependencies.
All three backend libraries already declare `sarek_ir` in their `(libraries ...)`.
Adding `Sarek_ir_codegen` to `sarek_ir` introduces no new dependency edges.

New module signature (`spoc/ir/Sarek_ir_codegen.ml` + `.mli`):

```ocaml
val mangle_name : string -> string
(** Replace '.' with '_' for C/GLSL identifier compatibility. *)

val gen_variant_def :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  constructor_prefix:string ->
  Buffer.t ->
  string * (string * Sarek_ir_types.elttype list) list ->
  unit
(** Emit C enum + typedef struct + union + constructor functions for one variant type.
    Used by CUDA, OpenCL, Metal backends. *)

val gen_variant_def_glsl :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  Buffer.t ->
  string * (string * Sarek_ir_types.elttype list) list ->
  unit
(** Emit GLSL const-int enum + struct (flat fields, no union) + constructor functions.
    Used by Vulkan backend. *)
```

Call site transformations:

```ocaml
(* CUDA — generate_with_types line 881 *)
List.iter
  (Sarek_ir_codegen.gen_variant_def
     ~type_of_elttype:cuda_type_of_elttype
     ~constructor_prefix:"__device__ __host__ inline"
     buf)
  k.kern_variants

(* OpenCL — generate_with_types line 859 *)
List.iter
  (Sarek_ir_codegen.gen_variant_def
     ~type_of_elttype:opencl_type_of_elttype
     ~constructor_prefix:"static inline"
     buf)
  k.kern_variants

(* Metal — see Known Issue; either add this or delete the dead gen_variant_def *)
List.iter
  (Sarek_ir_codegen.gen_variant_def
     ~type_of_elttype:metal_type_of_elttype
     ~constructor_prefix:"static inline"
     buf)
  k.kern_variants

(* Vulkan — generate_with_types line 1055 *)
List.iter
  (Sarek_ir_codegen.gen_variant_def_glsl
     ~type_of_elttype:glsl_type_of_elttype
     buf)
  k.kern_variants
```

All four local `mangle_name` definitions are deleted. All call sites qualified as
`Sarek_ir_codegen.mangle_name` (full qualification, no local alias).

## Quality Gates

```bash
# Build (use local opam switch)
opam exec --switch=/home/mathias/dev/SPOC -- dune build

# Tests (PPX + unit; GPU not required)
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest

# Format (ocamlformat via dune)
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote

# License header check
./scripts/check-license-headers.sh
```

## Open Questions

_(empty — everything resolved by code reading)_
