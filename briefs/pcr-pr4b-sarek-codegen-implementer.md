# Implementer Sub-brief — pure-codegen-rollout PR-4b (extract sarek_codegen)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-4 of 5, part b)
**Type:** refactor (structural; pure — NO behavior/codegen change)

## Goal
Extract the 4 backend code generators (`Sarek_ir_cuda`, `Sarek_ir_opencl`, `Sarek_ir_metal`,
`Sarek_ir_glsl`) — currently each living inside its ctypes-FFI backend package — into a NEW
**pure `sarek_codegen` library** with deps only `sarek_ir`, `sarek_registry`,
`sarek_backend_error` (all ctypes-free as of PR-1..4a). Each backend package then RE-EXPORTS its
`Sarek_ir_<b>` so existing consumers are unchanged. This is the capstone that makes
`OCaml IR → backend source` compile with zero FFI — the enabler for PR-5's FFI-free bytecode/jsoo.

**Pure refactor: generated source must be BYTE-IDENTICAL.** `sarek/tests/codegen_golden/` is the
oracle. Any golden diff = regression → STOP.

## Evidence (already verified — do NOT re-investigate)
- Each generator's ONLY `Spoc_core` reference is `generate_for_device ~(device : Spoc_core.Device.t)`
  (cuda l.825, opencl l.808, metal l.1034; glsl has none). This function is **DEAD** — no caller
  anywhere (only stale `gh-pages/` HTML + doc/error strings reference it; no `.mli`, plugin, test,
  or benchmark calls it). Nothing sets `current_framework` to `Some`, so the SNative `None` branch
  (which errors) is already the only reachable path — removing `generate_for_device` is behavior-neutral.
- Generators call `<Backend>_error.{raise_error,unknown_intrinsic,unsupported_construct,invalid_arg_count}`.
  `Cuda_error` etc. are just `include Sarek_backend_error.Backend_error.Make(struct let name="CUDA" end)`
  (+ `exception Cuda_error = …Backend_error`). The raised exception is the shared `Backend_error`.
- Generators call `Sarek_registry.fun_device_template` (now pure) and `Sarek_pure_registry.*` (pure).
- Plugins call `Sarek_ir_<b>.generate` / `.generate_with_types` (unqualified, in-package).
  `benchmarks/generate_backend_code.ml` calls `Sarek_cuda.Sarek_ir_cuda.generate` (wrapped path).
- Shared `Sarek_ir_codegen` already lives in `sarek_ir` (`spoc/ir/`); leave it there.

## Steps (build + goldens green after EACH; COMMIT after each step on the branch)
1. **Per generator, remove dead `generate_for_device`** and any now-unused `Spoc_core` reference.
   Keep `current_framework`/`current_variants` refs and the SNative match AS-IS (byte-identical).
   Confirm each generator source no longer references `Spoc_core`. Build + goldens green. COMMIT.
2. **Per generator, replace `<Backend>_error`** with a locally-instantiated module at the top:
   `module Codegen_error = Sarek_backend_error.Backend_error.Make (struct let name = "CUDA" end)`
   (resp. "OpenCL"/"Metal"/"Vulkan"), and rewrite `Cuda_error.X` → `Codegen_error.X`. The raised
   exception type is unchanged (shared `Backend_error`). Build + goldens green. COMMIT.
3. **Create `sarek_codegen`** (`sarek/codegen/` or a new dir — your call): move the 4 generator
   `.ml` (and any `.mli`) there. `(library (name sarek_codegen) (public_name sarek.codegen)
   (libraries sarek_ir sarek_registry sarek_backend_error) (modules Sarek_ir_cuda Sarek_ir_opencl
   Sarek_ir_metal Sarek_ir_glsl) (preprocess no_preprocessing))`. MUST build with NO ctypes/
   spoc_core/spoc_framework dep. Build + goldens green. COMMIT.
4. **Re-export from each backend package** so consumers are unchanged: in `sarek-cuda/`, replace
   the moved source with a thin re-export module `Sarek_ir_cuda.ml` = `include
   Sarek_codegen.Sarek_ir_cuda` (so `Sarek_cuda.Sarek_ir_cuda.*` and in-package unqualified
   `Sarek_ir_cuda.*` still resolve), add `sarek_codegen` to the backend's `(libraries …)`, drop
   the moved module from the FFI lib's source set. Same for opencl/metal/vulkan. Build all 4
   plugins + `benchmarks`. Build + goldens green. COMMIT.
5. **Golden shims** (`sarek/tests/codegen_golden/`): they currently `(select … from (sarek_cuda ->
   real)(-> stub))`. Prefer repointing `codegen_golden_backends` to depend DIRECTLY on
   `sarek_codegen` (always available — no select needed), so the harness tests pure codegen with no
   FFI link. If that's more churn than keeping the select, keep the select but ensure it still
   resolves. Either way goldens BYTE-IDENTICAL. Build + goldens green. COMMIT.
6. **Final gates.** COMMIT format fixes.

## Hard constraints
- **BYTE-IDENTICAL goldens** — `@sarek/tests/runtest`; `git diff origin/main -- sarek/tests/codegen_golden/` empty. Any diff = STOP.
- `sarek_codegen`'s `(libraries …)` = exactly `sarek_ir sarek_registry sarek_backend_error` — NO
  ctypes/ctypes.foreign/spoc_core/spoc_framework. This purity is the PR's whole point; verify it builds standalone.
- Consumers UNCHANGED: `Cuda/Opencl/Metal_plugin.ml`, `Vulkan_plugin.ml`, and `benchmarks/` build
  without source edits to their `Sarek_ir_<b>.*` call sites (the re-export preserves the paths).
- NO `Sarek_transpile` (PR-5). NO generator logic change (only: delete dead fn, swap error module, move files).
- SPDX headers on new files; `dune fmt` clean — and COMMIT any fmt changes (do not leave dirty).
- **COMMIT after each step on `phase0b/pr4b-sarek-codegen`; do NOT leave changes uncommitted.**

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek_codegen           # pure lib links, no ctypes
opam exec --switch=/home/mathias/dev/SPOC -- dune build                         # all plugins + benchmarks
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune exec sarek/tests/e2e/test_float32_sin_pure.exe -- --vulkan
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed .ml/dune>
```
(`-lnvrtc` full-build link error pre-existing; CI e2e-fast matrix-mul segfault known-flaky.)
