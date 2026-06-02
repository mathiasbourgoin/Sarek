# Implementer Sub-brief — pure-codegen-rollout PR-2 (pure registry + dual-registration + sinf)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-2 of 5)

## Goal
Make the intrinsic registry usable from the pure (no-`Device.t`) path across the whole
stdlib, so the generators can resolve intrinsics without `Spoc_core`. Land the **`sinf`
fix** (path-qualified `Float32` math emits the single-precision name) — a deliberate,
intake-approved behavior change — with golden updates and GPU validation. Native/interpreter
behaviour stays identical.

## Decisions already made (do NOT re-litigate)
- **ADOPT `sinf`** for path-qualified `Float32` math (and `cosf/sqrtf/expf/...`, plus the
  OpenCL/Metal/GLSL equivalents). `main` wrongly emits double `sin` for a `float32` op.
- Pure device closure type: `framework:string -> string` (no `Spoc_core.Device.t`).

## Reference
The Phase 0A spike branch `phase0a/pure-codegen-spike` holds a proven `Sarek_pure_registry.ml`
(in `sarek_ir`, ctypes-free) + the CUDA `gen_intrinsic` pure-registry query + a
`float32_sin_pure` golden. Get it with `git show phase0a/pure-codegen-spike:spoc/ir/Sarek_pure_registry.ml`.
Promote and generalize it.

## Steps (build + goldens green after each; COMMIT after each step on your worktree branch)
1. **Add `Sarek_pure_registry`** to `sarek_ir` (from the spike). It maps
   `(module-path, name) -> (framework:string -> string)` with zero `Device.t`.
2. **Populate it for the whole stdlib** — `Float32/Float64/Int32/Int64/Math/Gpu` intrinsics,
   types, consts. Prefer making the `%sarek_intrinsic` PPX (`sarek/ppx_intrinsic/`) emit a
   **pure** registration (`framework:string->string`) alongside the existing FFI one — so
   the two registries stay in sync from one definition. If PPX dual-registration proves too
   large/risky in a bounded attempt, fall back to a build-time-generated static pure table
   and SAY SO (escalate the scope, don't force a half-migration).
3. **All four generators** (`Sarek_ir_{cuda,opencl,metal}.ml`, `Sarek_ir_glsl.ml`) query the
   pure registry first for path-qualified intrinsics (`path <> []`), mirroring the spike's
   CUDA change. Drop the dead unqualified pure-registry entries the spike reviewer flagged.
4. **Adopt `sinf` + update goldens.** Regenerate ONLY the goldens that legitimately change
   due to the `sin→sinf` (and siblings) fix; in the PR, the golden diff must be exactly the
   intended `f`-suffix (and analogous) changes — no unexpected drift. Add golden kernels
   covering path-qualified `Float32` math for all 4 backends.
5. **GPU-validated e2e.** Add/extend an e2e that runs a `Float32.sin` (or similar) kernel and
   checks numerical correctness; confirm it passes on the Vulkan backend (RX 7900 XTX):
   `dune exec sarek/tests/e2e/<test>.exe -- --vulkan`.
6. **Native/interpreter unchanged** — they keep the FFI registry/runtime; confirm their
   tests still pass.

## Hard constraints
- Native/interpreter output identical; only the GPU path-qualified math output changes (to the `f` form), and ONLY where intended — every golden change must be an expected `sinf`-class change, justified in the report.
- No library split (PR-3), no `sarek_codegen` extraction (PR-4), no `Sarek_transpile` (PR-5).
- SPDX headers; `dune fmt` clean.
- **COMMIT your work on the worktree branch after each step** (do not leave changes uncommitted).

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens reflect intended sinf changes only
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune exec sarek/tests/e2e/<float32-test>.exe -- --vulkan
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed .ml>
```
(`-lnvrtc` full-build error pre-existing; CI e2e-fast matrix-mul segfault known-flaky.)
