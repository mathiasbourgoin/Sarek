# Implementation Report — PR-4b: Extract sarek_codegen

**Branch:** `phase0b/pr4b-sarek-codegen`
**Status:** COMPLETE — all gates passed

## Commits

```
356156da style(codegen): step 6 — ocamlformat + SPDX headers (PR-4b)
b4a7097d refactor(tests): step 5 — repoint golden harness to sarek_codegen (PR-4b)
e04dfaa2 refactor(codegen): step 4 — re-export Sarek_ir_* from backend packages (PR-4b)
4492e766 feat(codegen): step 3 — create sarek_codegen pure library (PR-4b)
40c9bd6c refactor(codegen): step 2 — swap backend error modules for local Codegen_error (PR-4b)
c8da4a7f refactor(codegen): step 1 — remove dead generate_for_device (PR-4b)
5dc87dfc style(ppx,tests): apply ocamlformat to ppx and unit test files (preamble)
```

## Files Modified / Created

### New
- `sarek/codegen/dune` — pure library definition, `(libraries sarek_ir sarek_registry sarek_backend_error)`
- `sarek/codegen/Sarek_ir_cuda.ml` — CUDA generator (moved from `sarek-cuda/`)
- `sarek/codegen/Sarek_ir_opencl.ml` — OpenCL generator (moved from `sarek-opencl/`)
- `sarek/codegen/Sarek_ir_metal.ml` — Metal generator (moved from `sarek-metal/`)
- `sarek/codegen/Sarek_ir_glsl.ml` — GLSL/Vulkan generator (moved from `sarek-vulkan/`)

### Modified (backend re-exports)
- `sarek-cuda/Sarek_ir_cuda.ml` — replaced with `include Sarek_codegen.Sarek_ir_cuda`
- `sarek-opencl/Sarek_ir_opencl.ml` — replaced with `include Sarek_codegen.Sarek_ir_opencl`
- `sarek-metal/Sarek_ir_metal.ml` — replaced with `include Sarek_codegen.Sarek_ir_metal`
- `sarek-vulkan/Sarek_ir_glsl.ml` — replaced with `include Sarek_codegen.Sarek_ir_glsl`
- `sarek-cuda/dune`, `sarek-opencl/dune`, `sarek-metal/dune`, `sarek-vulkan/dune` — added `sarek_codegen` to `(libraries ...)`

### Modified (golden test harness)
- `sarek/tests/codegen_golden/dune` — removed `(select ...)` stanzas, direct `sarek_codegen` dep with `(rule (copy ...))` stanzas
- `sarek/tests/codegen_golden/gen_{cuda,opencl,metal,glsl}.real.ml` — `open Sarek_codegen` instead of `open Sarek_{cuda,opencl,metal,vulkan}`

### Side-effect (pre-existing SPDX gaps fixed)
- `sarek-metal/Metal_api.ml`, `sarek-metal/Metal_types.ml` — added missing `SPDX-FileCopyrightText`
- `sarek/tests/e2e/backend_metal.available.ml`, `backend_metal.unavailable.ml` — same

## Re-export Approach

Each backend replaces its full generator source with a 5-line stub:
```ocaml
include Sarek_codegen.Sarek_ir_cuda
```
This preserves both consumer paths:
- Unqualified in-package: `Sarek_ir_cuda.*` (within `sarek-cuda`)
- Wrapped: `Sarek_cuda.Sarek_ir_cuda.*` (external consumer)

## Error Module Decision

Each generator now has a locally-instantiated `Codegen_error` module:
```ocaml
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "CUDA"
end)
```
This eliminates the only FFI-adjacent dependency in each generator. The raised
exception type (`Backend_error.Backend_error`) is identical to what
`Cuda_error`/etc. raise — no exception type change at the boundary.

## Gate Results

| Gate | Result |
|------|--------|
| `@sarek/tests/runtest` goldens byte-identical | PASS (20/20 each step) |
| `git diff origin/main -- sarek/tests/codegen_golden/test_codegen_golden.ml` empty | PASS |
| `sarek_codegen (libraries ...)` = `sarek_ir sarek_registry sarek_backend_error` | PASS |
| `git grep 'Spoc_core\|Ctypes\|Spoc_framework'` on 4 generator sources | CLEAN |
| `dune build sarek/codegen/` — no ctypes link | PASS |
| `dune build @sarek-vulkan/all` | PASS |
| `dune build` (full, pre-existing -lnvrtc failure only) | PASS |
| `test_float32_sin_pure.exe --vulkan` on RX 7900 XTX | PASS |
| ocamlformat clean | PASS |
| SPDX headers | PASS |

## Backend Status

- CUDA: re-exported; full generator in `sarek_codegen`; `-lnvrtc` link failure pre-existing (no CUDA hardware), not introduced here
- OpenCL: re-exported; full generator in `sarek_codegen`; builds fine
- Metal: re-exported; full generator in `sarek_codegen`; Metal not available in this env (Linux), not tested at e2e level
- Vulkan: re-exported; full generator in `sarek_codegen`; `@sarek-vulkan/all` builds; `test_float32_sin_pure --vulkan` PASSES on RX 7900 XTX
