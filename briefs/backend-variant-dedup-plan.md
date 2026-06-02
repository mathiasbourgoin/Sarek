# Plan — backend-variant-dedup

**Date:** 2026-06-02
**Status:** VALIDATED

## Sequential steps

1. **Add `Sarek_ir_codegen` to `spoc/ir/dune`** — one-word change to `(modules ...)`.
   Files: `spoc/ir/dune`.
   Completion: dune recognises the module name.

2. **Create `spoc/ir/Sarek_ir_codegen.ml` and `spoc/ir/Sarek_ir_codegen.mli`** with
   `mangle_name`, `gen_variant_def` (C/MSL), and `gen_variant_def_glsl` (GLSL).
   Add SPDX CECILL-B license headers to both files before writing.
   The C body is taken verbatim from `Sarek_ir_cuda.ml:796–868` with `cuda_type_of_elttype`
   replaced by `type_of_elttype ty` and the prefix string replaced by `constructor_prefix`.
   The GLSL body is taken verbatim from `Sarek_ir_glsl.ml:969–1036` with
   `glsl_type_of_elttype` replaced by `type_of_elttype ty`.
   Files: `spoc/ir/Sarek_ir_codegen.ml`, `spoc/ir/Sarek_ir_codegen.mli`.
   Completion: `dune build` succeeds with the new module in scope.

3. **Migrate CUDA** — delete local `mangle_name` (line 33) and `gen_variant_def`
   (lines 796–868); update all `mangle_name` call sites to
   `Sarek_ir_codegen.mangle_name`; update call site in `generate_with_types` (line 881)
   to use `Sarek_ir_codegen.gen_variant_def ~type_of_elttype:cuda_type_of_elttype
   ~constructor_prefix:"__device__ __host__ inline"`.
   Files: `sarek-cuda/Sarek_ir_cuda.ml`.
   Completion: `dune build` clean.

4. **Migrate OpenCL** — same pattern; `opencl_type_of_elttype`, prefix `"static inline"`.
   Call site: line 859.
   Files: `sarek-opencl/Sarek_ir_opencl.ml`.
   Completion: `dune build` clean.

5. **Migrate Metal** — delete local `mangle_name` and `gen_variant_def`; update
   `mangle_name` call sites. Then resolve the dead-code finding:
   - Read `sarek-metal/test/test_sarek_ir_metal.ml` for any variant-type test.
   - If a variant codegen test exists and currently passes without a `List.iter` call:
     the gap is known and intentional — do NOT add the call; note in a code comment.
   - If no such test exists or the gap looks like an oversight: add the missing
     `List.iter (Sarek_ir_codegen.gen_variant_def ~type_of_elttype:metal_type_of_elttype
     ~constructor_prefix:"static inline" buf) k.kern_variants` call to
     `generate_with_types` after the record defs block.
   Files: `sarek-metal/Sarek_ir_metal.ml`.
   Completion: `dune build` clean, decision documented.

6. **Migrate Vulkan** — delete local `mangle_name` and `gen_variant_def`; update call
   site at line 1055 to `Sarek_ir_codegen.gen_variant_def_glsl
   ~type_of_elttype:glsl_type_of_elttype`.
   Files: `sarek-vulkan/Sarek_ir_glsl.ml`.
   Completion: `dune build` clean.

7. **Full build + tests**
   ```bash
   opam exec --switch=/home/mathias/dev/SPOC -- dune build
   opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
   ```
   Completion: all tests pass.

8. **Format + license**
   ```bash
   opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote
   ./scripts/check-license-headers.sh
   ```
   Completion: clean diff after fmt; license check exits 0.

## Dependencies

- Step 2 depends on step 1 (dune must list the module).
- Steps 3–6 depend on step 2 (module must be compilable).
- Steps 3–6 are independent of each other (different files); do sequentially with a
  `dune build` check between each to isolate failures.
- Steps 7–8 depend on steps 3–6.

## Identified risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Metal dead `gen_variant_def` — decision wrong | Medium | Low | Test suite is the oracle; if no variant test exists, add a TODO comment, don't silently add a call |
| Missed `mangle_name` call site | Low | Low (compile-time error) | OCaml catches unbound values; incremental per-file build |
| License header check fails on new files | Low | Low | Add SPDX headers before running the check, not after |
| Partial application type error at call site | Low | Low (compile-time error) | Labelled args + positional `buf` — verified correct for `List.iter` |
| No variant codegen regression test | Medium | Medium | Note as gap; covered by reviewer |

## Decisions made

| Point | Decision | Reason |
|---|---|---|
| Vulkan scope | Included with `gen_variant_def_glsl` | GLSL target structurally different; needs its own function |
| `mangle_name` style | Full qualification `Sarek_ir_codegen.mangle_name` | Explicit dependency, no hidden alias |
| Metal dead code | Implementer decides based on test suite | Cannot determine intent without reading Metal tests |
| `generate_with_types` | Out of scope | Backend-specific structure, not safely parameterisable |
| `.mli` for `Sarek_ir_codegen` | Required | Public `(wrapped false)` library — explicit interface prevents accidental leakage |

## Assumptions

- No `.mli` files exist for the four backend IR modules (so no interface files need updating).
- `spoc/ir/test/dune` does not need updating (no tests added in this refactor).
- The three C-backend `gen_variant_def` bodies are byte-for-byte equivalent in generated
  output (confirmed by reading; only `type_of_elttype` fn and prefix differ).
- `sarek_ir` library does not transitively depend on any backend — adding `Sarek_ir_codegen`
  introduces no circular dependency.
