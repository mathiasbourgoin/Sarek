# Core Runtime

## Component Inventory

Reviewed core files: `sarek/core/README.md`, `Advanced.ml`, `Device.ml`, `Error.ml`, `Gpu_memory.ml`, `Kernel.ml`, `Kernel_arg.ml`, `Log.ml`, `Memory.ml`, `Profiling.ml`, `Runtime.ml`, `Transfer.ml`, `Vector.ml`, `Vector_storage.ml`, `Vector_transfer.ml`, `Vector_types.ml`, `dune`, and colocated tests under `sarek/core/test/**`.

## Per-File Purpose

- `sarek/core/README.md`: design overview, vector transfer state machine, device/runtime concepts, and test list.
- `sarek/core/Vector_types.ml`: shared vector element kind and metadata types.
- `sarek/core/Vector_storage.ml`: host arrays, sub-vector metadata, buffer bookkeeping, and logical vector storage.
- `sarek/core/Vector_transfer.ml`: transfer-facing helpers layered around vector storage.
- `sarek/core/Vector.ml`: public vector operations, host accessors, initialization, blit, fill, gather, and conversions.
- `sarek/core/Device.ml`: device identity, backend kind, capabilities, and properties.
- `sarek/core/Memory.ml`: memory-size helpers and memory-info records.
- `sarek/core/Transfer.ml`: CPU/GPU/Both/Stale_CPU location transitions and backend buffer movement.
- `sarek/core/Gpu_memory.ml`: GPU allocation accounting, retry helpers, finalizers, and transfer counters.
- `sarek/core/Kernel_arg.ml`: typed kernel argument encoding.
- `sarek/core/Kernel.ml`: abstract kernel modules, argument setters, and launch interface.
- `sarek/core/Runtime.ml`: backend registration/selection facade and kernel-loading helpers.
- `sarek/core/Error.ml`: shared error variants and formatting.
- `sarek/core/Advanced.ml`: futures, streams, events, unified-memory placeholders, and graph-like APIs.
- `sarek/core/Profiling.ml`: profiling events and counters.
- `sarek/core/Log.ml`: runtime logging helpers.
- `sarek/core/dune`: core library build and dependencies.

## Features/APIs

- Host vectors with typed element sizes and metadata.
- Device descriptors for CUDA, OpenCL, CPU, and native-like backends.
- Explicit vector transfers through `Transfer.to_device`, `Transfer.to_cpu`, `mark_cpu_modified`, and `mark_gpu_modified`.
- Kernel abstraction with typed scalar/vector arguments.
- Memory accounting hooks and retry-on-allocation-failure support.
- Logging and profiling utilities for runtime integration.
- Placeholder advanced APIs for async, streams, unified memory, and graph capture.

## Invariants

- `Vector_storage.location` must always describe where the authoritative data lives.
- `Both d` means host and device data for `d` are synchronized.
- `Stale_CPU d` means device data is authoritative and the host copy is stale.
- Moving a vector from one device to another must first read from the authoritative copy.
- Freeing a device buffer must not discard newer data unless the API explicitly invalidates the vector.
- Sub-vectors must preserve offset, length, and element size relative to the parent vector.
- Kernel argument values must match the backend's expected order and type.
- (PR #168, branch `feature/spoc-core-decouple`) The pure numeric vector core `spoc_core_base` must NOT link `ctypes` or `unix`; this is enforced at build time by `sarek/core/ffi_free_gate/` (a Bigarray `Vector.create/set/get/to_array` program that builds to bytecode AND js_of_ocaml and runs under `node`). If `ctypes`/`unix` re-enter the numeric core, the gate fails to build.

## Potential Invariant Violations/Bugs

- `Vector_storage.sub_vector_host` creates sub-vector metadata and host storage at `sarek/core/Vector_storage.ml:214-258`, but the public `Vector.gather` path is a placeholder/no-op around sub-vector synchronization at `sarek/core/Vector.ml:282-296`. This makes partition/gather semantics incomplete for device-backed vectors.
- Possible overlapping blit corruption: `Vector.blit` copies forward only at `sarek/core/Vector.ml:501-517`. If source and destination are the same vector and ranges overlap, the behavior can differ from `Array.blit`. Uncertain: the public API may not promise overlap-safe blits.
- `Gpu_memory.with_retry` catches every exception and retries after cleanup at `sarek/core/Gpu_memory.ml:130-143`, not just allocation/OOM failures. This can hide programming errors or perform unrelated cleanup.
- `Advanced.Future` appears nonfunctional: `is_ready` may return true without storing `Ready`, `await` can fail at `sarek/core/Advanced.ml:93-95`, `map` does not apply the mapper at `sarek/core/Advanced.ml:97-107`, and `run_async` does not launch asynchronous work at `sarek/core/Advanced.ml:116-122`.

## Recently Resolved

- Cross-device transfer from `Stale_CPU old_dev` or `GPU old_dev` to a different device was fixed by PR #136, merged as `5dffea3`.
- `Transfer.free_buffer` and `Transfer.free_all_buffers` now preserve authoritative device data by synchronizing it before cleanup; this was fixed by PR #136.

## Web/JS decouple — merged to main (PRs #168–#171)

### OCaml/jsoo WebGPU pipeline (current state, all on `main`)

| Layer | Library | What it provides |
|---|---|---|
| `sarek/core_base/` | `spoc_core_base` | Pure `Make(CUSTOM_OPS)` functor, FFI-free numeric vector core |
| `sarek/core_js/` | `spoc_core_js` | `Make(Js_ops)` — browser numeric Vector API (runs in node/jsoo) |
| `sarek/core_js/webgpu/` | `spoc_webgpu_runtime` | `Webgpu_js` glue + `Webgpu_runtime.run` — explicit-layout WebGPU dispatch via jsoo |
| `sarek/core_js/webgpu/lessons/` | (driver exe) | `compose_driver.bc.js` — OCaml host composing two kernels; exported as `SpocCompose.run` |
| `sarek/plugins/webgpu/` | `sarek_webgpu` | Inert skeleton implementing `Framework_sig.BACKEND` (is_available=false, stubs) |
| Lesson #7 | `gh-pages/learn/07-compose.html` | First lesson powered by real OCaml runtime: two kernels in sequence |

**Verified on real RX 7900 XTX:** `make webgpu-runtime-test` (vector_add + sin) and `make compose-gpu-test` (a²+b composition) both ALL PASS.

**Design:** hidden-functor (`CUSTOM_OPS`), B1 async read-back seam, precompiled lessons (learner edits kernels, host OCaml is precompiled). The existing kernel-only lessons (01–06) stay on `sarek_webgpu_runner.js` (JS); new lessons use the OCaml runtime.

**Open follow-ups:** #24 native cutover (route native Vector through `Make(Ctypes_ops)`, remove duplication, split file); #25 compose-lesson WGSL panes; multi-output guard in `Webgpu_runtime`.

## Web/JS decouple — `spoc_core_base` (PR #168, merged)

To let SPOC host code run in the browser, the numeric (Bigarray) vector core is being split so it links **FFI-free** (compilable to js_of_ocaml), while native behaviour stays **byte-identical** and the public `Spoc_core.*` API is unchanged. Design: `docs/plans/spoc-web-backend-2026-06-04.md` (decision: **hidden functor**).

- **`sarek/core_base/Spoc_core_base.ml`** (+`.mli`, `dune`): new pure library (deps = `sarek_ir` only). `module Make (Ops : CUSTOM_OPS)` is the numeric vector core; `Custom_storage` carries `Ops.handle` instead of `unit Ctypes.ptr`. `CUSTOM_OPS` abstracts the custom-type handle + alloc/free/`add_offset`/`copy_elems`/`of_raw`/`to_raw`/`bigarray_to_handle`/`device_id`.
- **Native `sarek/core` (`spoc_core`)** keeps its name/public API and provides `Ctypes_ops` (`handle = unit Ctypes.ptr`); `Vector_types.ml` does `include Spoc_core_base.Make (Ctypes_ops)`. Because `Ctypes_ops` has no `.mli`, `handle` stays transparently `unit Ctypes.ptr`, so `custom_type.get/set` and `Custom_helpers.*` keep their exact signatures.
- **Shims:** `Memory.ml` `bigarray_elem_size` (pure lookup) replaces `Ctypes_static.sizeof`; `Profiling.clock` is an injectable `(unit -> float) ref` (default `Unix.gettimeofday`; jsoo overrides it).
- **Gate:** `sarek/core/ffi_free_gate/` (`Stub_ops` + `gate_numeric`) builds bytecode + `.bc.js` and runs under `node`.
- **Known follow-up (review Med-1, task #24):** native `Vector` still routes through the original `Vector_storage`/`Vector_transfer` rather than through `Make(Ctypes_ops)` — the functor is proven FFI-free (via the gate) but the native host-create path is duplicated (two `next_id` counters; `Spoc_core_base.ml` > 500-line limit). Cutover/dedup pending. `Ctypes_ops.free` is a no-op (GC-managed `allocate_n`, matches pre-refactor); `Profiling.clock` is unsynchronized global mutable state (set-once intended).

## Performance Or Maintainability Risks

- Transfer state is spread across `Vector_storage`, `Vector_transfer`, `Vector`, and `Transfer`, increasing the chance that future changes update one path but not another.
- Global finalizer/accounting behavior in `Gpu_memory` may make tests order-dependent unless counters are carefully reset.
- Catch-all retry in `Gpu_memory` can produce extra cleanup work and make failures harder to diagnose.
- Advanced APIs are broad but placeholder-like; users can build against semantics that are not implemented.
- Device capability fields exist, including FP64 support, but enforcement is mostly outside core and may be inconsistent.

## Related Tests

- `sarek/core/test/test_device.ml`: device construction and properties.
- `sarek/core/test/test_vector.ml`: vector creation and host operations.
- `sarek/core/test/test_vector_storage.ml`: storage metadata and host storage behavior.
- `sarek/core/test/test_vector_transfer.ml`: vector transfer helper behavior.
- `sarek/core/test/test_memory.ml`: memory info and sizes.
- `sarek/core/test/test_kernel.ml`: kernel abstraction tests.
- `sarek/core/test/test_kernel_arg.ml`: kernel argument encoding.
- `sarek/core/test/test_gpu_memory.ml`: memory accounting and retry basics.

`sarek/core/README.md:829` references `test_transfer.ml`, but the colocated test inventory currently has `test_vector_transfer.ml` instead.

## Missing Tests

- Sub-vector partition, device update, gather, and parent consistency.
- Overlapping `Vector.blit` with same source and destination.
- Retry behavior that distinguishes backend OOM from non-OOM exceptions.
- Advanced future/stream/event behavior, or explicit tests that these APIs are placeholders.

## Concrete Improvement/Fix Candidates

- Replace `Vector.blit` internals with overlap-aware direction selection or delegate to typed host-array `Array.blit`.
- Implement real sub-vector gather semantics or clearly mark sub-vector device support unsupported.
- Narrow `Gpu_memory.with_retry` to backend memory-allocation exceptions.
- Move placeholder advanced APIs behind explicit unsupported errors until semantics are implemented.
