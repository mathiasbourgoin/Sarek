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
- `Sarek_ir_metal.ml`: **stale — verified 2026-07-02.** The in-package `sarek-metal/Sarek_ir_metal.ml` is now a 9-line re-export stub (`include Sarek_codegen.Sarek_ir_metal`); the real MSL generator (scalar/vector parameters, arrays, helpers, records, variants, atomics, thread IDs, synchronization) lives in `sarek/codegen/Sarek_ir_metal.ml`. Variant/struct emission (`gen_variant_def`, `mangle_name`) delegates to the shared `Sarek_ir_codegen` module in `spoc.ir` from there (`sarek/codegen/Sarek_ir_metal.ml:36`, `1089-1090`). See [README](README.md).

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

**Caveat on all Metal entries in this pass: SOURCE-VERIFIED ONLY.** No macOS/Metal hardware access was available; every "fixed" classification below is based on reading `sarek-metal/*.ml` and matching it against the commit log (`49da4768..618768b7`), not on running the code on real Metal hardware.

- **fixed 2026-07-02 (merged, source-verified only):** `Metal_api.ml` previously defined a local `exception Metal_error of string` separate from the structured `Metal_error` module, so runtime failures (e.g. allocation failure) raised a different exception shape than the structured backend errors. `Metal_api.check` now raises the canonical `Metal_error.Metal_error` (a `Backend_error` alias) via the shared `Sarek_backend_error.Backend_error.Make.check` funnel (`sarek-metal/Metal_api.ml:39-`); the old local exception is kept only as an `[@ocaml.deprecated]` alias (`Metal_api.ml:26-34`, updated text: `"no longer raised; Metal_api.check now raises Metal_error.Metal_error (Backend_error) - catch that instead"`) for opam-published out-of-tree compatibility. Payload changed too — the canonical path stringifies the Metal result via `to_string` before wrapping, so catchers get a formatted string, not a typed value; see `kb/backends/shared-patterns.md` for the cross-backend payload note.
- Device `get 0` returns the system default device while `count` may use `MTLCopyAllDevices`; index 0 can mean different handles depending on path at `sarek-metal/Metal_api.ml:122-135`. The NSArray from device enumeration is not released.
- **Kept live, verified 2026-07-02 (source-verified only):** `supports_fp64` is explicitly false at `sarek-metal/Metal_api.ml:110-118`, while `generate_with_fp64` (`sarek/codegen/Sarek_ir_metal.ml:1177`) is just an alias for `generate` and does not add FP64 support; the implementation still maps `TFloat64` to `float`. This contradiction is unchanged by the #213/#214 pass — explicitly kept live per this audit's "Metal fp64 contradiction" instruction.
- Buffer allocation computes `size * elem_size` without negative-size or overflow validation at `sarek-metal/Metal_api.ml:175-187`. Not addressed by this pass.
- Host/device copies do not validate both endpoint capacities at `sarek-metal/Metal_plugin_base.ml:253-279`. **Still open** — kept live per the copy-size-validation-gap instruction; no fix found.
- `device_to_device` silently copies the minimum of source and destination sizes at `sarek-metal/Metal_plugin_base.ml:281-288`; truncation can hide caller bugs. Not addressed by this pass — kept live per the "device_to_device copy-size validation gaps" instruction.
- `alloc_zero_copy` returns `None` and `is_zero_copy` returns false even though comments say shared memory is effectively zero-copy at `sarek-metal/Metal_plugin_base.ml:241-245`. Not addressed by this pass.
- Event timing is wall-clock placeholder behavior at `sarek-metal/Metal_plugin_base.ml:312-325`; it does not use GPU timestamps. **Still open** — kept live per the "Event wall-clock stubs" instruction.
- **fixed 2026-07-02 (merged, source-verified only):** argument setters previously stored indices (`set_arg_*`) but `launch` discarded them and mapped args sequentially by `List.iteri`, so out-of-order setting bound wrong Metal buffers/scalars. All setters now route through the shared `Spoc_framework.Kernel_args` container (`type args = arg Spoc_framework.Kernel_args.t`, `Metal_plugin_base.ml:222`), and `launch` extracts via `Kernel_args.validate_and_extract` keyed on the caller-supplied `idx` (`Metal_plugin_base.ml:265-307`).
- **KEEP LIVE — arity-validation gap, documented in code (verified 2026-07-02, source-verified only):** same nuance as CUDA/Native — Metal compiled-kernel handles carry no arity metadata, so `expected_count` falls back to `Kernel_args.count args` (the number of distinct indices set), per an explicit code comment: `"KNOWN GAP: Metal compiled-kernel handles carry no arity metadata, so -- as with Native/CUDA -- expected_count falls back to the number of distinct indices actually set. This still rejects internal gaps/duplicates but cannot catch a caller that consistently omits a trailing argument."` (`Metal_plugin_base.ml:296-300`). Gaps/duplicates/out-of-range are caught; pure trailing under-count is not. Intentional, not closed by this pass.
- **Reclassified 2026-07-02 (source-verified only), not a live bug:** `Metal_api.Kernel.execute` (`sarek-metal/Metal_api.ml:311-337`) still binds Metal buffer/scalar args by `List.iteri` position over the `arg list` it receives — but this is no longer evidence of the same bug, because `Metal_plugin_base.launch` (the only caller reached through the framework) now builds that list from `Kernel_args.validate_and_extract`, which is already ordered by the caller-supplied `idx`, not call order. `execute`'s `List.iteri` here is just the mechanism for turning a correctly-pre-ordered list into sequential `[[buffer(N)]]` FFI calls — that ordering is exactly what Metal's ABI expects. Direct callers of `Metal_api.Kernel.execute` that bypass `Metal_plugin_base` and hand-build an `arg list` in the wrong order would still be able to misbind, but that is a "trust the caller" contract at the low-level API layer, not the previously-documented framework-level bug.
- `supported_source_langs` advertises `OpenCL_Source` while the backend compiles source as Metal at `sarek-metal/Metal_plugin.ml:194-200`.
- `generate_source` ignores the requested block size at `sarek-metal/Metal_plugin.ml:174-181`; launch uses block size for dispatch, but generated source does not encode or validate expected threadgroup dimensions.
- `metal_atomic_type_of_elttype` emits `atomic_float` for `TFloat32` while the comment says Metal does not support atomic float at `sarek/codegen/Sarek_ir_metal.ml:72-76` (path updated 2026-07-02). Marked likely compile issue.
- Generic `atomic_add` always casts to `volatile threadgroup atomic_int*` at `sarek/codegen/Sarek_ir_metal.ml:384-406`; device memory atomics require the separate `atomic_add_global_int32` path at `sarek/codegen/Sarek_ir_metal.ml:408-428`. A generic atomic on a device vector likely emits the wrong address space.
- `atomic_sub`, `atomic_min`, and `atomic_max` emit names such as `atomic_sub(...)` at `sarek/codegen/Sarek_ir_metal.ml:433-474`; these may not be valid MSL builtins in this form. Marked uncertain pending compiler check.
- `SWarpBarrier` emits `sub_group_threadgroup_barrier` at `sarek/codegen/Sarek_ir_metal.ml:724-729`, while the plugin intrinsic registry uses `simdgroup_barrier` for subgroup barrier semantics. Marked likely codegen/API drift.
- RESOLVED (2026-06-02): `generate_with_types` previously emitted record definitions but not variant definitions, so variant use with typed generation missed required type/constructor definitions. It now emits variant typedefs by calling the shared `gen_variant_def` before record definitions, keeping Metal consistent with the CUDA/OpenCL C-family backends (`sarek/codegen/Sarek_ir_metal.ml:1114`; `gen_variant_def` delegates to `Sarek_ir_codegen.gen_variant_def` at `sarek/codegen/Sarek_ir_metal.ml:1089-1090`; paths updated 2026-07-02).
- `Metal_bindings.ml` uses multiple typed `objc_msgSend` signatures and `objc_msgSend_stret` for `MTLSize` at `sarek-metal/Metal_bindings.ml:171-181`; this is ABI-sensitive across x86_64 and arm64. Marked high-risk FFI issue.
- NSString objects created for source/function names are not released after library/function creation in `sarek-metal/Metal_bindings.ml:263-291`.
- **New entry, verified 2026-07-02 (source-verified only) — do not close:** `generate_source` catches all exceptions and returns `None` at `sarek-metal/Metal_plugin.ml:181` (`with _ -> None`), hiding codegen failure reasons, matching the same pattern already tracked for OpenCL/Vulkan. Confirmed still present against current source; not touched by #213/#214.

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

- Replace `supported_source_langs = [OpenCL_Source]` with a Metal-specific source language if available, or reject external source until the framework has one. Still open.
- ~~Preserve argument indices through `Metal_api.Kernel.arg`, sort/validate before launch, and test sparse/out-of-order arguments.~~ **DONE 2026-07-02 (merged, source-verified only)** — via shared `Kernel_args` in `Metal_plugin_base.ml`.
- ~~Remove the local `Metal_api.Metal_error` exception or convert it at API boundaries into structured `Metal_error` values.~~ **DONE 2026-07-02 (merged, source-verified only)** — `Metal_api.check` now raises canonical `Metal_error.Metal_error` (Backend_error); old exception kept only as a deprecated compatibility alias.
- Add size/overflow validation to allocation and all copy functions; make device-to-device size mismatch an error. **Still open** — copy-size validation gap kept live per this audit's instructions.
- Emit variant definitions in `generate_with_types`. (DONE 2026-06-02 — `generate_with_types` now calls `gen_variant_def`.)
- Compile generated MSL in tests on macOS CI, covering generic/device atomics and subgroup barriers. Still open.
- Release Objective-C temporary strings and audited Objective-C objects after use. Still open.
- Migrate the Metal compile-cache key onto `Compile_cache.make_key`. **DONE 2026-07-02 (merged, source-verified only)** — `Metal_plugin_base.ml:233-241` (commit `940436ac`).
