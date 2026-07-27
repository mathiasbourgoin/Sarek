# Sarek New Runtime Tests

## Component Inventory

New runtime tests live in `sarek/tests/new_runtime/`.

## Per-File Purpose

- `sarek/tests/new_runtime/dune`: declares `test_native_runtime` and CUDA-gated `test_runtime_comparison`.
- `sarek/tests/new_runtime/test_native_runtime.ml`: tests native plugin initialization, Sarek type registration, PPX kernel generation, manual native kernel registration, and `Execute.run_vectors`.
- `sarek/tests/new_runtime/test_runtime_comparison.ml`: compares runtime execution across CUDA, OpenCL, Native, and Interpreter where available.

## Features And APIs

- `test_native_runtime` is declared with `sarek_native` and `sarek_ppx` preprocessing in `sarek/tests/new_runtime/dune:4-9`.
- `test_runtime_comparison` links CUDA/OpenCL libraries and is enabled only when `CUDA_PATH` is non-empty in `sarek/tests/new_runtime/dune:11-20`.
- `test_native_runtime.ml` initializes the native plugin at `sarek/tests/new_runtime/test_native_runtime.ml:36`.
- It registers a Sarek record type at `sarek/tests/new_runtime/test_native_runtime.ml:42-45`.
- It runs generated/runtime kernels through `Execute.run_vectors` at `sarek/tests/new_runtime/test_native_runtime.ml:225`.
- `test_runtime_comparison.ml` defines a PPX vector-add kernel at `sarek/tests/new_runtime/test_runtime_comparison.ml:27-37` and invokes `Sarek.Execute.run_vectors` at `sarek/tests/new_runtime/test_runtime_comparison.ml:66`.

## Invariants

- Native plugin initialization must make at least one native device available for native-runtime checks.
- Generated kernels should expose IR before runtime execution; both new runtime tests fail if `kernel.ir` is missing.
- Runtime comparison should only require GPU libraries when the Dune gate says they are present.

## Potential Invariant Violations Or Bugs

- Confirmed: this directory is not part of top-level default `runtest` in `sarek/tests/dune:1-15`.
- **Corrected 2026-07-02 (was overstated as a build/link risk):** `test_runtime_comparison` is gated only by `CUDA_PATH` in `sarek/tests/new_runtime/dune:17-20` while linking `sarek_cuda`/`sarek_opencl`, but the OpenCL-link concern is mitigated — NVRTC and the CUDA driver are loaded lazily via `Dl.dlopen` at runtime (see `Cuda_nvrtc.ml`/`Cuda_bindings.ml`), not linked statically (`sarek/tests/new_runtime/dune:19-22`, comment in the dune file itself). `sarek_opencl`'s own bindings follow the same pattern (`sarek-opencl/Opencl_bindings.ml:10,29-41`: "All bindings are lazy — they only dlopen the library when first used", trying `libOpenCL.so.1`/`.so`/`.dylib`/`OpenCL.dll` in turn). So an environment with `CUDA_PATH` set but no OpenCL library present is not expected to fail at `dune build` or link time from either side; a missing library only surfaces as a runtime error when `test_runtime_comparison` actually attempts to use that backend.
- Confirmed maintainability risk: tests use `failwith` and printed status messages rather than Alcotest assertions, for example `sarek/tests/new_runtime/test_native_runtime.ml:112-123` and `sarek/tests/new_runtime/test_runtime_comparison.ml:44`.

## Performance Or Maintainability Risks

- GPU/runtime comparison is environment-sensitive and not part of default feedback.
- Manual native kernel registration in the test can drift from PPX-generated native fallback behavior.
- Non-Alcotest reporting makes failures less structured in CI logs.

## Related Tests

- Native codegen unit tests are listed in `sarek/tests/unit/dune:14-15`.
- E2E native/debug/external kernel declarations are in `sarek/tests/e2e/dune:183-196`.

## Missing Tests

- A default native-runtime smoke test that does not require CUDA/OpenCL.
- Structured assertions for missing native device, missing IR, and incorrect results.
- Separate CUDA and OpenCL gates for runtime comparison.

## Concrete Improvement/Fix Candidates

- Add a top-level gated alias for new runtime tests.
- Split `test_runtime_comparison` into backend-specific executables or gates.
- Convert printed failures and `failwith` branches into Alcotest checks.
