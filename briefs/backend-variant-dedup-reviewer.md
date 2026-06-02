# Reviewer Brief — backend-variant-dedup

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## What was implemented

A new shared module `Sarek_ir_codegen` was added to `spoc/ir/` (the `sarek_ir` library).
It exports `mangle_name`, `gen_variant_def` (C/MSL), and `gen_variant_def_glsl` (GLSL).
Four backend IR files (`Sarek_ir_cuda.ml`, `Sarek_ir_opencl.ml`, `Sarek_ir_metal.ml`,
`Sarek_ir_glsl.ml`) had their local copies of these functions deleted and their call
sites updated to use the shared module.

This is a pure refactor — no functional change to generated GPU code is intended.

## Audit checklist

### 1. `spoc/ir/Sarek_ir_codegen.ml` — correctness of extracted bodies

- [ ] `mangle_name` is `String.map (fun c -> if c = '.' then '_' else c) name` — identical to all four originals.
- [ ] `gen_variant_def` body matches `Sarek_ir_cuda.ml:796–868` with `cuda_type_of_elttype` replaced by `type_of_elttype` and prefix string replaced by `constructor_prefix`. No other change.
- [ ] `gen_variant_def_glsl` body matches `Sarek_ir_glsl.ml:969–1036` with `glsl_type_of_elttype` replaced by `type_of_elttype`. No other change.
- [ ] Both new files carry SPDX `CECILL-B` license headers.

### 2. Call sites — semantic equivalence

For each backend, verify that the new call site produces identical buffer content to the original:

| Backend | Original call | New call | Equivalent? |
|---|---|---|---|
| CUDA | `gen_variant_def buf v` (local) | `Sarek_ir_codegen.gen_variant_def ~type_of_elttype:cuda_type_of_elttype ~constructor_prefix:"__device__ __host__ inline" buf v` | ✓/✗ |
| OpenCL | `gen_variant_def buf v` (local) | `Sarek_ir_codegen.gen_variant_def ~type_of_elttype:opencl_type_of_elttype ~constructor_prefix:"static inline" buf v` | ✓/✗ |
| Metal | (was dead) | (either added or noted) | N/A |
| Vulkan | `gen_variant_def buf v` (local) | `Sarek_ir_codegen.gen_variant_def_glsl ~type_of_elttype:glsl_type_of_elttype buf v` | ✓/✗ |

### 3. `mangle_name` call sites — completeness

For each backend, verify no remaining use of the (now-deleted) local `mangle_name`:

```bash
grep -n '\bmangle_name\b' sarek-cuda/Sarek_ir_cuda.ml sarek-opencl/Sarek_ir_opencl.ml \
  sarek-metal/Sarek_ir_metal.ml sarek-vulkan/Sarek_ir_glsl.ml
```

Every remaining occurrence must be `Sarek_ir_codegen.mangle_name`. Zero unqualified uses allowed.

### 4. Metal dead-code decision

- [ ] The implementer made an explicit decision (comment or added call).
- [ ] If a call was added: it matches the pattern for OpenCL (same `static inline` prefix, `metal_type_of_elttype`).
- [ ] If a comment was added: it explains the gap and references the brief.
- [ ] No silent deletion without documentation.

### 5. Scope discipline

- [ ] `generate_with_types` functions are unchanged (except the `gen_variant_def` call site line).
- [ ] `*_type_of_elttype` functions are unchanged except for `mangle_name` → `Sarek_ir_codegen.mangle_name`.
- [ ] No other functions moved or modified.
- [ ] `spoc/ir/test/dune` unchanged (no new test dependency added).

### 6. dune file

- [ ] `spoc/ir/dune` has `Sarek_ir_codegen` in `(modules ...)`.
- [ ] No new `(libraries ...)` entries in any backend dune file (no new dep edges).

### 7. Quality gates passed

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build           # must pass
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest  # must pass
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote  # must produce no further changes
./scripts/check-license-headers.sh                                 # must exit 0
```

## Risks to specifically check

- **Vulkan `gen_variant_def_glsl` output equivalence**: the GLSL version uses `Printf.sprintf` throughout; any substitution error silently changes generated GLSL source. Compare the body character-by-character against the original if there is any doubt.
- **`mangle_name` inside recursive `type_of_elttype`**: missed call sites inside the recursive type function bodies would cause a compile-time unbound-value error — but only if `dune build` was run after each file. Check the build logs.
- **No variant codegen regression test**: if no test exercises a kernel with a variant type, a silent output difference would not be caught. Flag this gap in the review verdict.

## Expected verdict

**GO** if all checklist items pass and quality gates are clean.
**NO-GO** if: any unqualified `mangle_name` remains, any call site is semantically different from its original, or any quality gate fails.
