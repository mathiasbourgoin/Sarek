# CUDA Backend

## Component Inventory

- `sarek-cuda/README.md`: package documentation, examples, and test descriptions.
- `sarek-cuda/dune`: optional `sarek-cuda.plugin` library; modules are declared at `sarek-cuda/dune:7-28`.
- `sarek-cuda/Cuda_error.ml`: shared backend error helpers.
- `sarek-cuda/Cuda_types.ml`: CUDA handles, enums, result conversion, device properties, dimensions.
- `sarek-cuda/Cuda_bindings.ml`: dynamic `libcuda` loading and CUDA Driver API FFI.
- `sarek-cuda/Cuda_nvrtc.ml`: dynamic NVRTC loading and CUDA C to PTX compilation.
- `sarek-cuda/Cuda_api.ml`: device, memory, stream/event, and kernel APIs.
- `sarek-cuda/Cuda_plugin_base.ml`: SPOC framework backend implementation.
- `sarek-cuda/Cuda_plugin.ml`: backend registration, intrinsic registry, source generation, and external CUDA source execution.
- `sarek-cuda/Sarek_ir_cuda.ml`: Sarek IR to CUDA C code generator.
- `sarek-cuda/test/`: `test_cuda_error.ml`, `test_sarek_ir_cuda.ml`, and test `dune`.
- `sarek-cuda/CHANGELOG.md`: package change notes.

## Per-File Purpose

- `Cuda_error.ml` instantiates shared `Backend_error.Make`; most runtime API functions instead raise the local `Cuda_api.Cuda_error` exception defined in `sarek-cuda/Cuda_api.ml:25-31`.
- `Cuda_types.ml` defines `cu_result`, opaque handles, memory copy structs, `dim3`, and conversion helpers. `string_of_cu_result` collapses many values to `CUDA_ERROR_OTHER` at `sarek-cuda/Cuda_types.ml:420-436`.
- `Cuda_bindings.ml` lazily loads `libcuda` from common Linux/macOS paths at `sarek-cuda/Cuda_bindings.ml:39-68`; profiler calls are optional and become no-ops if missing at `sarek-cuda/Cuda_bindings.ml:497-525`.
- `Cuda_nvrtc.ml` loads `libnvrtc` at `sarek-cuda/Cuda_nvrtc.ml:99-117`, maps compute capabilities to `compute_XX`, and exposes `compile_to_ptx`.
- `Cuda_api.ml` initializes the driver, caches devices, allocates/copies memory, manages streams/events, compiles source to PTX, caches kernels, and launches kernels.
- `Cuda_plugin_base.ml` adapts `Cuda_api` to the framework signatures, including memory, streams, events, kernel argument accumulation, and launch.
- `Cuda_plugin.ml` exposes CUDA as a JIT backend with registration priority 100 and environment-based disable checks at `sarek-cuda/Cuda_plugin.ml:227-256`.
- `Sarek_ir_cuda.ml` handles CUDA type mapping, intrinsic mapping, expression/statement generation, helper functions, records, variants, and kernel signatures.

## Features and APIs

- Driver API execution with lazy `libcuda` loading.
- Runtime CUDA C compilation through NVRTC.
- Device enumeration, context creation, memory allocation, host/device/device copies, streams, events, PTX module loading, kernel cache, and kernel launch.
- CUDA source is the only supported external source language at `sarek-cuda/Cuda_plugin.ml:171`.
- Code generation supports scalar/vector parameters, arrays, records, variants, shared locals, synchronization, common math intrinsics, and CUDA thread intrinsics.

## Invariants

- `Cuda_api.init` must run before device operations; it is guarded by `initialized` in `sarek-cuda/Cuda_api.ml:51-60`.
- A cached `Device.t` context should remain valid while it is returned by `Device.get`.
- Kernel cache keys must distinguish every semantic input that changes the loaded function.
- Kernel argument setters must bind the user-supplied argument index, not just append order.
- Copy byte counts must fit both source and destination buffers.
- NVRTC programs and loaded CUDA modules should be destroyed on every failure path after allocation.

## Potential Invariant Violations and Bugs

- `sarek-cuda/Cuda_error.ml:22` defines `let module_load_failed ptx_size reason = module_load_failed ptx_size reason`. This appears self-recursive or at least ambiguous after the `include Error`; it should explicitly delegate to the included helper or be removed. Marked likely bug.
- `Device.destroy` destroys the CUDA context but leaves the entry in `device_cache` at `sarek-cuda/Cuda_api.ml:164-178`; a later `Device.get` can return a destroyed context.
- Allocation computes `size * elem_size` without negative-size or overflow validation at `sarek-cuda/Cuda_api.ml:191-205`.
- Copy APIs do not validate the opposite side capacity. `host_to_device`, `device_to_host`, and `device_to_device` use one side's byte count at `sarek-cuda/Cuda_api.ml:211-238`.
- Kernel cache keys omit `name`; `compile_cached` keys only on device id and source digest at `sarek-cuda/Cuda_api.ml:399-406`. A source containing multiple kernels can return the wrong cached `CUfunction`.
- `Cuda_plugin_base.Kernel.set_arg_*` ignores the supplied `idx` and appends to a list at `sarek-cuda/Cuda_plugin_base.ml:268-281`; out-of-order setting or replacement binds wrong arguments and append is O(n).
- `Cuda_nvrtc.compile_to_ptx` destroys the program on compile failure and success, but if `nvrtcGetPTXSize` or `nvrtcGetPTX` fails after a successful compile, the program can leak at `sarek-cuda/Cuda_nvrtc.ml:379-387`.
- `Sarek_ir_cuda.EVariant` emits `make_` plus the raw type name at `sarek-cuda/Sarek_ir_cuda.ml:156-158`, while variant constructors are generated with the mangled type name at `sarek-cuda/Sarek_ir_cuda.ml:841-842`. Type names containing `.` likely produce mismatched or invalid CUDA.
- Nullary variant expressions emit the bare constructor tag at `sarek-cuda/Sarek_ir_cuda.ml:155`, which may not be a value of the enclosing variant struct. Marked likely bug pending typed IR examples.
- `atomic_add` supports two or three args but reports expected count `3` in the error path at `sarek-cuda/Sarek_ir_cuda.ml:293-312`.

## Performance and Maintainability Risks

- `initialized`, `device_cache`, and kernel cache refs are unsynchronized. Concurrent initialization, destroy, or compile can race.
- `Cuda_api` uses a local exception instead of the structured `Cuda_error` module, so callers see two different CUDA error surfaces.
- README module sizes and API descriptions are stale. For example, `Cuda_error.ml` is documented as much larger than the current 22-line file, and several listed API names do not correspond to the current modules.
- The NVRTC architecture fallback clamps capabilities above 9.0 to `compute_90` at `sarek-cuda/Cuda_api.ml:318-327`; this is conservative but may leave newer hardware under-targeted until updated.

## Related Tests

- `sarek-cuda/test/dune:3-15` defines Alcotest suites with bisect instrumentation.
- `sarek-cuda/test/test_cuda_error.ml:118-137` covers shared error constructors and formatting.
- `sarek-cuda/test/test_sarek_ir_cuda.ml:253-277` covers literals, basic operations, statements, declarations, control flow, and helper pieces of code generation.

## Missing Tests

- Runtime test for `Device.destroy` followed by `Device.get`.
- Kernel cache test with the same CUDA source containing two kernel names.
- Out-of-order and repeated `set_arg_*` tests.
- Allocation/copy overflow and bounds tests.
- NVRTC cleanup test for post-compile failure paths.
- Full generated CUDA C compile tests for records, variants, shared memory, atomics, and vector length parameters.

## Concrete Improvement Candidates

- Include `name` in the CUDA kernel cache key and add a regression test with two kernels in one source.
- Store kernel args by index, validate contiguous required arguments before launch, and support replacement.
- Remove stale `device_cache` entries in `Device.destroy`, or make contexts process-lifetime and document that `destroy` is not supported.
- Wrap NVRTC program lifetime in `Fun.protect` after creation.
- Add byte-size validation helpers shared by `Memory.host_to_device`, `device_to_host`, and `device_to_device`.
- Fix variant constructor mangling and add a generated-code test for namespaced variants.
