# Shared Patterns and Tests

## Shared Architecture

The GPU backends are optional packages that adapt backend-specific runtime APIs to the SPOC framework. Each backend has:

- a shared error wrapper (`*_error.ml`);
- backend-specific FFI and type layers (`*_bindings.ml`, `*_types.ml`);
- a higher-level runtime API (`*_api.ml`);
- a framework adapter (`*_plugin_base.ml`); CUDA/OpenCL/Metal share `module type Framework_sig.PLUGIN_BASE`;
- a registration/source-generation layer (`*_plugin.ml`);
- an IR code generator, now split across two layers per backend (see [README](README.md)): a 9-line `Sarek_ir_*.ml` re-export stub inside each backend package, and the real generator in the `sarek_codegen` library (`sarek/codegen/Sarek_ir_*.ml`), which itself delegates shared variant/struct emission to `Sarek_ir_codegen` in `spoc.ir` (`spoc/ir/Sarek_ir_codegen.ml`);
- two main test suites: one for error helpers and one for codegen fragments.

## Resolved Duplication (2026-06-02, paths refreshed 2026-07-02 after the codegen-extraction move)

- Variant/struct codegen is no longer duplicated across backends. `gen_variant_def` and `mangle_name` were previously copied into each `Sarek_ir_<backend>.ml`; they now delegate to the shared `Sarek_ir_codegen` module in the `spoc.ir` library. This logic lives in `sarek/codegen/` since the 2026-07-02 codegen extraction (the in-package `sarek-*/Sarek_ir_*.ml` files are now pure re-export stubs and carry no line-level content of their own). CUDA, OpenCL, and Metal call `Sarek_ir_codegen.gen_variant_def` (`sarek/codegen/Sarek_ir_cuda.ml:831`, `sarek/codegen/Sarek_ir_opencl.ml:815`, `sarek/codegen/Sarek_ir_metal.ml:1090`); Vulkan calls `Sarek_ir_codegen.gen_variant_def_glsl` (`sarek/codegen/Sarek_ir_glsl.ml:1047`). `mangle_name` is aliased to `Sarek_ir_codegen.mangle_name` in each (`sarek/codegen/Sarek_ir_cuda.ml:36`, `sarek/codegen/Sarek_ir_opencl.ml:47`, `sarek/codegen/Sarek_ir_metal.ml:36`, `sarek/codegen/Sarek_ir_glsl.ml:38`).
- The CUDA, OpenCL, and Metal `*_plugin_base.ml` modules no longer each carry a ~130-line inline backend signature; they now share `module type Framework_sig.PLUGIN_BASE` (`spoc/framework/Framework_sig.ml:362-493`). Each plugin base module is annotated `: Framework_sig.PLUGIN_BASE` (`sarek-cuda/Cuda_plugin_base.ml:15`, `sarek-opencl/Opencl_plugin_base.ml:17`, `sarek-metal/Metal_plugin_base.ml:17`). Vulkan is not part of this dedup: `Vulkan_plugin_base.ml` never carried an inline signature (`sarek-vulkan/Vulkan_plugin_base.ml:16`).
- Latent Metal bug fixed alongside the codegen extraction (intentional, reviewed): `Sarek_ir_metal.ml`'s `generate_with_types` previously never emitted variant typedefs, unlike the other C-family backends; it now calls `gen_variant_def` before record definitions (`sarek/codegen/Sarek_ir_metal.ml:1114`, generate_with_types at `:1097`). This is the only behavior change in the 2026-06-02 refactor pass; all other changes are pure code moves. (Path updated 2026-07-02 — code lives in `sarek/codegen/` post codegen-extraction, not `sarek-metal/`.)

## Audit-Verified Status Matrix (2026-07-02, superseded same-day by the #213/#214 fix pass)

Historical snapshot, cross-checked against code as of `49da4768`:

| Invariant | CUDA | OpenCL | Vulkan | Metal |
|---|---|---|---|---|
| Copy-size validation (both endpoints) | violated (`Cuda_api.ml:236-239`) | violated (`Opencl_api.ml:437`) | violated (`Vulkan_api_memory.ml:409-543`) | violated (`Metal_plugin_base.ml:157-164`, also truncates on d2d) |
| Kernel/arg indexing honored | violated — appended, idx ignored (`Cuda_plugin_base.ml:150-163`) | mostly honored, but duplicate-index replacement applies the *older* value (`Opencl_plugin_base.ml:299-323`) | violated — sequential bindings, idx ignored (`Vulkan_api_kernel.ml:351-354`) | violated — idx stored but discarded at launch (`Metal_plugin_base.ml:381-393`/`406-416`) |
| Cache key includes kernel/entry-point name | violated (`Cuda_api.ml:432-440`) | not violated | violated (`Vulkan_api_kernel.ml:314-320`) | not violated |

**fixed 2026-07-02 (merged)** — both rows above are now closed for all four backends, source-verified against `618768b7` (PR #213 + review fixes). Evidence:

- **Kernel/arg indexing**: a new shared container, `spoc/framework/Kernel_args.ml` (`type 'a t = {slots : (int, 'a) Hashtbl.t}`, `set`/`count`/`validate_and_extract`), stores args by caller-supplied `idx` (last-write-wins) and validates a contiguous `0..expected_count-1` range before launch (missing/extra index detection, `Kernel_args.ml:31-68`). All six backends now route through it: `Cuda_plugin_base.ml:136-180`, `Opencl_plugin_base.ml:215-309`, `Metal_plugin_base.ml:222-304`, `Vulkan_api_kernel.ml:55-114` (split buffer/scalar stores plus `validate_buffer_indices` for negative/out-of-count rejection), `sarek/plugins/native/Native_plugin_base.ml:565-757`, `sarek/plugins/interpreter/Interpreter_plugin_base.ml:528-680`.
- **Cache key includes name**: a shared `spoc/framework/Compile_cache.make_key ~device ~name ~source ?options ()` (`Compile_cache.ml:35-42`) independently digests each component (including `name`) before joining, closing the collision risk. CUDA uses it at `Cuda_api.ml:498-500`; Vulkan at `Vulkan_api_kernel.ml:385-390`; Metal migrated its compile-cache key to it too (commit `940436ac`, `Metal_plugin_base.ml`). Vulkan's on-disk `Framework_cache` additionally bumped its key schema to `"v2"` specifically to add the kernel/entry name to the digest (`sarek/framework/Framework_cache.ml:90-94`, comment: "v1 keys omitted it, so two kernels sharing one source string collided").

**Still open, unaffected by this pass:** copy-size validation (both endpoints) remains violated in all four backends — no evidence of a fix in `49da4768..618768b7`; keep this row live.

**New (2026-07-02):** nine identically-named functions are independently reimplemented across all four backends — `check`, `is_disabled`, `generate_source`, `generate_with_types`, `register_intrinsic`, `find_intrinsic`, `is_available`, `init`, `registered_backend` (e.g. `Cuda_api.ml:30`/`Cuda_shared.ml:126`/`Cuda_plugin.ml:11-25`, `Opencl_api.ml:22`/`Opencl_plugin.ml:267-297`, `Vulkan_api_base.ml:26`/`Vulkan_plugin.ml:266-317`, `Metal_plugin_base.ml:319`/`Metal_plugin.ml:240-272`). **fixed 2026-07-02 (merged)** — the "two distinct exceptions share one name per backend" half of this is now closed: `check` in each backend's `*_api.ml` (`Cuda_api.ml:39-`, `Opencl_api.ml:30-`, `Metal_api.ml:39-`, `Vulkan_api_base.ml:34-`) now raises the canonical `Backend_error` (via `Sarek_backend_error.Backend_error.Make.check`) instead of the old per-backend local exception. The 9x `check`/`is_disabled`/etc. reimplementation duplication itself is still open — no shared dispatcher was introduced, only the error-raising path was unified.

### Canonical error funnel — payload change (fixed 2026-07-02, merged)

`Backend_error.Make.check ~is_success ~to_string ctx result` (`spoc/framework_error/Backend_error.ml`) now stringifies the raw backend result via `to_string result` before wrapping it in `context_error`, rather than passing the raw error code/variant through as part of the exception payload (as the old per-backend exceptions did, e.g. `Cuda_error of cu_result * string`). Concretely this means: catching the canonical exception now gets a formatted string reason, not a typed `cu_result`/`cl_error`/`vk_result` value — code that pattern-matched on the old exception's structured payload needs to switch to string matching/parsing if it still needs the original code. The old per-backend exceptions (`Cuda_api.Cuda_error`, `Opencl_api.Opencl_error`, `Metal_api.Metal_error`, `Vulkan_api_base.Vk_result_error`) still exist as declarations, each carrying `[@ocaml.deprecated "no longer raised; ... raises ...Error (Backend_error) - catch that instead"]`, kept only for opam-published out-of-tree compatibility.

## Shift-semantics note (2026-07-02, verified)

Codegen for `Shr` in all four C-family/GLSL text backends (`Sarek_ir_cuda.ml:228`, `Sarek_ir_opencl.ml:246`, `Sarek_ir_metal.ml:248`, `Sarek_ir_glsl.ml:347`) still emits plain `" >> "`, unchanged — these were already correct (arithmetic shift on a signed operand is what C/MSL/GLSL `>>` does). The actual fix (commit `87864852`, `fix(ppx/lower-ir): give lsr logical-shift semantics, canonicalize Shr to arithmetic`) was in the two backends that previously disagreed: PTX (`shr.u32` → `shr.s32`) and the interpreter (`Int32.shift_right_logical` → `Int32.shift_right`), plus a new width-aware logical-shift IR expansion for `Lsr` in `Sarek_lower_ir.ml` built from existing `Shr`/`Shl`/`BitXor`/`Eq`/`EIf` nodes (no new IR constructor, since `formal`/`codegen-ptx` model `Shr` and are untouchable in that task). Net effect: `Shr` is now consistently arithmetic across every backend that consumes the shared IR; do not read this as "the text backends changed" — they did not need to.

## Shared Invariants

- Dynamic libraries must be loaded lazily and errors should include the attempted candidates.
- FFI object lifetimes must be paired: created programs, modules, buffers, command resources, strings, and compiler objects need release on success and failure paths.
- Device caches must not return destroyed handles.
- Kernel cache keys must include source, backend/device, entry point, and any compile options that affect generated code.
- Kernel arguments must be indexed and validated, not merely appended.
- Memory transfer helpers must validate byte counts against both source and destination.
- Codegen should either emit valid backend language for a construct or raise a structured unsupported-construct error.
- README examples should be treated as executable documentation and kept in sync with actual module APIs.

## Test Coverage Observed

- CUDA has bisect-instrumented error and codegen tests in `sarek-cuda/test/dune:3-15`.
- OpenCL has error and codegen tests in `sarek-opencl/test/dune:1-5`.
- Vulkan has error and codegen tests in `sarek-vulkan/test/dune:3-9`.
- Metal has error and codegen tests in `sarek-metal/test/dune:3-9`.

The codegen tests mostly assert small snippets or generated string contents:

- CUDA: `sarek-cuda/test/test_sarek_ir_cuda.ml:253-277`.
- OpenCL: `sarek-opencl/test/test_sarek_ir_opencl.ml:192-216`.
- Vulkan: `sarek-vulkan/test/test_sarek_ir_glsl.ml:219-243`.
- Metal: `sarek-metal/test/test_sarek_ir_metal.ml:199-222`.

The shared error tests cover constructors, prefixes, and exception helpers:

- CUDA: `sarek-cuda/test/test_cuda_error.ml:118-137`.
- OpenCL: `sarek-opencl/test/test_opencl_error.ml:137-156`.
- Vulkan: `sarek-vulkan/test/test_vulkan_error.ml:143-162`.
- Metal: `sarek-metal/test/test_metal_error.ml:119-137`.

## Shared Gaps

- No generated-kernel compile tests for CUDA NVRTC, OpenCL compiler, Vulkan GLSL/SPIR-V, or Metal MSL.
- No runtime launch tests for actual devices or mocked FFI layers.
- No tests for out-of-order argument setting, sparse indices, or repeated argument replacement.
- No bounds/overflow tests for allocation and copy APIs.
- No tests for destroy/cache invalidation behavior.
- No resource cleanup tests for compile failures.
- Minimal or no tests for records, variants, shared memory, vector length parameters, FP64/int64 paths, subgroup/warp barriers, and backend-specific atomics.
- README examples are not tested, and several are out of sync with current APIs.

## Suggested Test Plan

1. Add pure unit tests for argument containers in all backends. These can run without GPU libraries.
2. Add codegen tests for representative full kernels: vector add, scalar params, records, variants, shared memory, and atomics.
3. Add optional toolchain compile tests gated by environment variables:
   - CUDA: NVRTC available.
   - OpenCL: platform compiler available.
   - Vulkan: shaderc or `glslangValidator` available.
   - Metal: macOS Metal compiler available.
4. Add negative tests that assert unsupported constructs raise structured backend errors rather than emitting comments or invalid code.
5. Add runtime integration tests behind opt-in flags for real-device execution and transfer validation.

## Cross-Backend Fix Candidates

- Introduce a small shared argument map abstraction with `set idx value`, replacement, contiguous validation, and backend-specific materialization.
- Introduce shared byte-size validation helpers for Bigarray/device-buffer copies.
- Add cache key helpers that standardize `(backend, device id/name, entry point, source digest, compile options)`.
- Split "backend unavailable" from "source compile failed" and preserve nested error messages for `generate_source` failures.
- Add doc tests or example compilation checks for README snippets after API stabilization.
