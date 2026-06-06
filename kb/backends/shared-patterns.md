# Shared Patterns and Tests

## Shared Architecture

The GPU backends are optional packages that adapt backend-specific runtime APIs to the SPOC framework. Each backend has:

- a shared error wrapper (`*_error.ml`);
- backend-specific FFI and type layers (`*_bindings.ml`, `*_types.ml`);
- a higher-level runtime API (`*_api.ml`);
- a framework adapter (`*_plugin_base.ml`); CUDA/OpenCL/Metal share `module type Framework_sig.PLUGIN_BASE`;
- a registration/source-generation layer (`*_plugin.ml`);
- an IR code generator (`Sarek_ir_*.ml`) that delegates shared variant/struct emission to `Sarek_ir_codegen` (`spoc.ir`);
- two main test suites: one for error helpers and one for codegen fragments.

## Resolved Duplication (2026-06-02)

- Variant/struct codegen is no longer duplicated across backends. `gen_variant_def` and `mangle_name` were previously copied into each `Sarek_ir_<backend>.ml`; they now delegate to the shared `Sarek_ir_codegen` module in the `spoc.ir` library. CUDA, OpenCL, and Metal call `Sarek_ir_codegen.gen_variant_def` (`sarek-cuda/Sarek_ir_cuda.ml:794-795`, `sarek-opencl/Sarek_ir_opencl.ml:776-777`, `sarek-metal/Sarek_ir_metal.ml:1003-1004`); Vulkan calls `Sarek_ir_codegen.gen_variant_def_glsl` (`sarek-vulkan/Sarek_ir_glsl.ml:968-969`). `mangle_name` is aliased to `Sarek_ir_codegen.mangle_name` in each (`sarek-cuda/Sarek_ir_cuda.ml:31`, `sarek-opencl/Sarek_ir_opencl.ml:42`, `sarek-metal/Sarek_ir_metal.ml:31`, `sarek-vulkan/Sarek_ir_glsl.ml:32`).
- The CUDA, OpenCL, and Metal `*_plugin_base.ml` modules no longer each carry a ~130-line inline backend signature; they now share `module type Framework_sig.PLUGIN_BASE` (`spoc/framework/Framework_sig.ml:362-493`). Each plugin base module is annotated `: Framework_sig.PLUGIN_BASE` (`sarek-cuda/Cuda_plugin_base.ml:15`, `sarek-opencl/Opencl_plugin_base.ml:17`, `sarek-metal/Metal_plugin_base.ml:17`). Vulkan is not part of this dedup: `Vulkan_plugin_base.ml` never carried an inline signature (`sarek-vulkan/Vulkan_plugin_base.ml:16`).
- Latent Metal bug fixed alongside the codegen extraction (intentional, reviewed): `Sarek_ir_metal.ml`'s `generate_with_types` previously never emitted variant typedefs, unlike the other C-family backends; it now calls `gen_variant_def` before record definitions (`sarek-metal/Sarek_ir_metal.ml:1028`). This is the only behavior change in the 2026-06-02 refactor pass; all other changes are pure code moves.

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
