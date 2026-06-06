# Metal Backend

## Component Inventory

- `sarek-metal/README.md`: package documentation and examples.
- `sarek-metal/dune`: optional `sarek-metal.plugin` library; modules are listed at `sarek-metal/dune:7-27`.
- `sarek-metal/Metal_error.ml`: shared backend error helpers.
- `sarek-metal/Metal_types.ml`: Objective-C/Metal handles, constants, resource options, dimensions, and error type.
- `sarek-metal/Metal_bindings.ml`: dynamic Metal/libobjc loading and Objective-C message FFI.
- `sarek-metal/Metal_api.ml`: higher-level device, buffer, command queue, library, pipeline, and execution APIs.
- `sarek-metal/Metal_plugin_base.ml`: framework backend implementation.
- `sarek-metal/Metal_plugin.ml`: registration, intrinsic registry, codegen, and external source execution.
- `sarek-metal/Sarek_ir_metal.ml`: Sarek IR to Metal Shading Language generator.
- `sarek-metal/test/`: `test_metal_error.ml`, `test_sarek_ir_metal.ml`, and test `dune`.

## Per-File Purpose

- `Metal_error.ml` instantiates `Backend_error.Make`.
- `Metal_types.ml` defines `id`, `SEL`, Metal object aliases, `MTLSize`, resource options, and a small `mtl_error` variant.
- `Metal_bindings.ml` opens the Metal framework and libobjc, allocates Objective-C selectors/strings, and binds selected device/buffer/library/pipeline/command encoder messages.
- `Metal_api.ml` wraps device enumeration, shared buffer allocation, copies, command queue creation, library/pipeline compilation, and synchronous kernel dispatch.
- `Metal_plugin_base.ml` adapts the API to framework memory, streams, events, cached kernels, argument lists, and launch; its `Metal` module is now constrained to the shared `Framework_sig.PLUGIN_BASE` module type instead of carrying an inline signature (`sarek-metal/Metal_plugin_base.ml:17`).
- `Metal_plugin.ml` provides Metal intrinsics and backend registration with priority 95.
- `Sarek_ir_metal.ml` emits MSL for scalar/vector parameters, arrays, helpers, records, variants, atomics, thread IDs, and synchronization. Variant/struct emission (`gen_variant_def`, `mangle_name`) now delegates to the shared `Sarek_ir_codegen` module in `spoc.ir` rather than carrying a duplicate inline implementation (`sarek-metal/Sarek_ir_metal.ml:31`, `1003-1008`).

## Features and APIs

- Runtime Metal compilation through `newLibraryWithSource`.
- Shared-storage buffers for CPU/GPU access.
- Command queues and synchronous command buffer execution.
- Codegen for Metal kernel signatures with `[[buffer(N)]]` arguments and built-in thread position arguments.
- Generated source from Sarek IR is Metal Shading Language. External-source advertising is currently incorrect: `supported_source_langs` returns `OpenCL_Source` at `sarek-metal/Metal_plugin.ml:194-196`, and `run_source` ignores `lang` at `sarek-metal/Metal_plugin.ml:198-200`.

## Invariants

- Objective-C FFI signatures must match platform ABI, especially for struct returns and typed `objc_msgSend`.
- Metal argument indices must match generated `[[buffer(N)]]` positions.
- Shared buffer copies must validate byte counts against both endpoints.
- MSL emitted by `Sarek_ir_metal.ml` must match actual Metal language features.
- Shared backend errors should be used consistently by runtime and plugin layers.

## Potential Invariant Violations and Bugs

- `Metal_api.ml` defines a local `exception Metal_error of string` at `sarek-metal/Metal_api.ml:26-34`, while the package also has the structured `Metal_error` module. Several low-level runtime failures raise the local exception rather than structured backend errors, for example allocation failure at `sarek-metal/Metal_api.ml:175-187`.
- Device `get 0` returns the system default device while `count` may use `MTLCopyAllDevices`; index 0 can mean different handles depending on path at `sarek-metal/Metal_api.ml:122-135`. The NSArray from device enumeration is not released.
- `supports_fp64` is explicitly false at `sarek-metal/Metal_api.ml:110-118`, while `generate_with_fp64` says Metal supports FP64 natively at `sarek-metal/Sarek_ir_metal.ml:1091`. The implementation maps `TFloat64` to `float`, so the comment/API story is inconsistent.
- Buffer allocation computes `size * elem_size` without negative-size or overflow validation at `sarek-metal/Metal_api.ml:175-187`.
- Host/device copies do not validate both endpoint capacities at `sarek-metal/Metal_plugin_base.ml:253-279`.
- `device_to_device` silently copies the minimum of source and destination sizes at `sarek-metal/Metal_plugin_base.ml:281-288`; truncation can hide caller bugs.
- `alloc_zero_copy` returns `None` and `is_zero_copy` returns false even though comments say shared memory is effectively zero-copy at `sarek-metal/Metal_plugin_base.ml:241-245`.
- Event timing is wall-clock placeholder behavior at `sarek-metal/Metal_plugin_base.ml:312-325`; it does not use GPU timestamps.
- Argument setters store indices at `sarek-metal/Metal_plugin_base.ml:381-393`, but `launch` discards them and maps args sequentially at `sarek-metal/Metal_plugin_base.ml:406-416`; out-of-order setting binds wrong Metal buffers/scalars.
- `Metal_api.Kernel.execute` also binds by `List.iteri` order at `sarek-metal/Metal_api.ml:302-337`.
- `supported_source_langs` advertises `OpenCL_Source` while the backend compiles source as Metal at `sarek-metal/Metal_plugin.ml:194-200`.
- `generate_source` ignores the requested block size at `sarek-metal/Metal_plugin.ml:174-181`; launch uses block size for dispatch, but generated source does not encode or validate expected threadgroup dimensions.
- `metal_atomic_type_of_elttype` emits `atomic_float` for `TFloat32` while the comment says Metal does not support atomic float at `sarek-metal/Sarek_ir_metal.ml:68-74`. Marked likely compile issue.
- Generic `atomic_add` always casts to `volatile threadgroup atomic_int*` at `sarek-metal/Sarek_ir_metal.ml:313-333`; device memory atomics require the separate `atomic_add_global_int32` path at `sarek-metal/Sarek_ir_metal.ml:334-358`. A generic atomic on a device vector likely emits the wrong address space.
- `atomic_sub`, `atomic_min`, and `atomic_max` emit names such as `atomic_sub(...)` at `sarek-metal/Sarek_ir_metal.ml:359-394`; these may not be valid MSL builtins in this form. Marked uncertain pending compiler check.
- `SWarpBarrier` emits `sub_group_threadgroup_barrier` at `sarek-metal/Sarek_ir_metal.ml:636-641`, while the plugin intrinsic registry uses `simdgroup_barrier` for subgroup barrier semantics. Marked likely codegen/API drift.
- RESOLVED (2026-06-02): `generate_with_types` previously emitted record definitions but not variant definitions, so variant use with typed generation missed required type/constructor definitions. It now emits variant typedefs by calling the shared `gen_variant_def` before record definitions, keeping Metal consistent with the CUDA/OpenCL C-family backends (`sarek-metal/Sarek_ir_metal.ml:1028`; `gen_variant_def` delegates to `Sarek_ir_codegen.gen_variant_def` at `sarek-metal/Sarek_ir_metal.ml:1003-1008`).
- `Metal_bindings.ml` uses multiple typed `objc_msgSend` signatures and `objc_msgSend_stret` for `MTLSize` at `sarek-metal/Metal_bindings.ml:171-181`; this is ABI-sensitive across x86_64 and arm64. Marked high-risk FFI issue.
- NSString objects created for source/function names are not released after library/function creation in `sarek-metal/Metal_bindings.ml:263-291`.

## Performance and Maintainability Risks

- `memcpy` is looked up inside the helper function on every call at `sarek-metal/Metal_api.ml:18-24`.
- Per-device state cache has no cleanup path at `sarek-metal/Metal_plugin_base.ml:153-172`.
- `max_threads_per_block` is computed as the product of reported width/height/depth at `sarek-metal/Metal_plugin_base.ml:189-218`; if those are per-dimension maxima, this can overstate total allowed threads. Marked uncertain.
- README limitations are stale: it says no device-to-device transfers at `sarek-metal/README.md:262-267`, but the plugin implements a memcpy-based `device_to_device` path at `sarek-metal/Metal_plugin_base.ml:281-288`.
- README contributor guidance says "No failwith - use `Metal_error.raise_error`" at `sarek-metal/README.md:301-306`, but the runtime still raises the local `Metal_api.Metal_error`.

## Related Tests

- `sarek-metal/test/dune:3-9` defines two Alcotest executables.
- `sarek-metal/test/test_metal_error.ml:119-137` covers shared error constructors and formatting.
- `sarek-metal/test/test_sarek_ir_metal.ml:199-222` covers literals, operations, simple statements, barriers, thread intrinsics, atomic snippets, type mapping, and helper declarations.

## Missing Tests

- MSL compile tests for generated kernels, especially atomics, barriers, records, variants, and float64.
- Runtime argument binding tests that set args out of order.
- External source language validation: Metal source accepted, OpenCL source rejected.
- Buffer copy bounds tests and device-to-device truncation behavior.
- Objective-C resource release tests or leak checks around NSString/library/function creation.
- Device enumeration consistency tests for multi-GPU Macs.
- GPU event timing or documented placeholder behavior.

## Concrete Improvement Candidates

- Replace `supported_source_langs = [OpenCL_Source]` with a Metal-specific source language if available, or reject external source until the framework has one.
- Preserve argument indices through `Metal_api.Kernel.arg`, sort/validate before launch, and test sparse/out-of-order arguments.
- Remove the local `Metal_api.Metal_error` exception or convert it at API boundaries into structured `Metal_error` values.
- Add size/overflow validation to allocation and all copy functions; make device-to-device size mismatch an error.
- Emit variant definitions in `generate_with_types`. (DONE 2026-06-02 — `generate_with_types` now calls `gen_variant_def`.)
- Compile generated MSL in tests on macOS CI, covering generic/device atomics and subgroup barriers.
- Release Objective-C temporary strings and audited Objective-C objects after use.
