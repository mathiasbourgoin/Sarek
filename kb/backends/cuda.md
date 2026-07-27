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
- `sarek-cuda/Cuda_shared.ml`: shared kargs extension, intrinsic registry, and `is_disabled`/`bind_args` helpers used by both CUDA plugins (`sarek-cuda/Cuda_shared.ml`, 143 lines).
- `sarek-cuda/Cuda_ptx_plugin.ml`: **updated 2026-07-02** — the default, auto-registered CUDA backend (priority 100). Emits PTX directly via `Sarek_ir_ptx` and loads it with `cuModuleLoadData` (no NVRTC). Cannot emit records/variants; `generate_source` returns `None` for those, and callers should fall back to `Cuda_c_plugin`.
- `sarek-cuda/Cuda_c_plugin.ml`: **new 2026-07-02** — the CUDA C backend (priority 90), NOT auto-registered by default (opt in via `Cuda_c_plugin.register ()`). Emits CUDA C via `Sarek_ir_cuda` and JIT-compiles through NVRTC; supports the full IR (records, variants, shared memory, device calls) that the PTX backend cannot yet emit.
- `sarek-cuda/Cuda_plugin.ml`: **reduced 2026-07-02 to a 25-line backward-compatibility shim.** It no longer contains registration logic itself; it forces `Cuda_ptx_plugin.init ()` and re-exports `register_intrinsic`/`find_intrinsic`/`generate_with_types`/`generate_source` from `Cuda_shared`/`Sarek_ir_cuda`.
- `sarek-cuda/Sarek_ir_cuda.ml`: 9-line re-export stub of `Sarek_codegen.Sarek_ir_cuda` (see [README](README.md) — codegen moved to the `sarek_codegen` library under `sarek/codegen/`).
- `sarek-cuda/test/`: `test_cuda_error.ml`, `test_sarek_ir_cuda.ml`, and test `dune`.
- `sarek-cuda/CHANGELOG.md`: package change notes.

## Per-File Purpose

- `Cuda_error.ml` instantiates shared `Backend_error.Make`; most runtime API functions instead raise the local `Cuda_api.Cuda_error` exception defined in `sarek-cuda/Cuda_api.ml:25-31`.
- `Cuda_types.ml` defines `cu_result`, opaque handles, memory copy structs, `dim3`, and conversion helpers. `string_of_cu_result` collapses many values to `CUDA_ERROR_OTHER` at `sarek-cuda/Cuda_types.ml:420-436`.
- `Cuda_bindings.ml` lazily loads `libcuda` from common Linux/macOS paths at `sarek-cuda/Cuda_bindings.ml:39-68`; profiler calls are optional and become no-ops if missing at `sarek-cuda/Cuda_bindings.ml:497-525`.
- `Cuda_nvrtc.ml` loads `libnvrtc` at `sarek-cuda/Cuda_nvrtc.ml:99-117`, maps compute capabilities to `compute_XX`, and exposes `compile_to_ptx`.
- `Cuda_api.ml` initializes the driver, caches devices, allocates/copies memory, manages streams/events, compiles source to PTX, caches kernels, and launches kernels.
- `Cuda_plugin_base.ml` adapts `Cuda_api` to the framework signatures, including memory, streams, events, kernel argument accumulation, and launch.
- `Cuda_shared.ml` defines the shared `Cuda_kargs` extension (so both plugins interoperate through the same wrap/unwrap), the shared `Cuda_intrinsics` registry, `bind_args`, and `is_disabled ()` — the environment-based disable check now lives here (`sarek-cuda/Cuda_shared.ml:126-128`: `SPOC_DISABLE_GPU` or `SPOC_DISABLE_CUDA`), not in `Cuda_plugin.ml`.
- `Cuda_ptx_plugin.ml` registers the `CUDA/PTX` backend at priority 100 (`sarek-cuda/Cuda_ptx_plugin.ml:82`), auto-registering on load unless `is_disabled ()`.
- `Cuda_c_plugin.ml` registers the `CUDA/C` backend at priority 90 (`sarek-cuda/Cuda_c_plugin.ml:84`); registration is explicit, not automatic.
- `Cuda_plugin.ml` is now a thin shim (25 lines) that forces `Cuda_ptx_plugin` to initialize and forwards a handful of names for backward compatibility.
- `Sarek_ir_cuda.ml` (in the `sarek_codegen` library, `sarek/codegen/Sarek_ir_cuda.ml`) handles CUDA type mapping, intrinsic mapping, expression/statement generation, helper functions, records, variants, and kernel signatures; the in-package `sarek-cuda/Sarek_ir_cuda.ml` is only a re-export.

## Features and APIs

- Driver API execution with lazy `libcuda` loading.
- Runtime CUDA C compilation through NVRTC (CUDA/C plugin only).
- Device enumeration, context creation, memory allocation, host/device/device copies, streams, events, PTX module loading, kernel cache, and kernel launch.
- Two JIT backends are registered, not one: `CUDA/PTX` (priority 100, auto-registered, `Framework_sig.PTX` source) and `CUDA/C` (priority 90, opt-in, `Framework_sig.CUDA_Source`). Each has its own `supported_source_langs`.
- Code generation supports scalar/vector parameters, arrays, records, variants, shared locals, synchronization, common math intrinsics, and CUDA thread intrinsics (CUDA/C path via `Sarek_ir_cuda`; the PTX path via `Sarek_ir_ptx` does not yet support records/variants).

## Invariants

- `Cuda_api.init` must run before device operations; it is guarded by `initialized` in `sarek-cuda/Cuda_api.ml:51-60`.
- A cached `Device.t` context should remain valid while it is returned by `Device.get`.
- Kernel cache keys must distinguish every semantic input that changes the loaded function.
- Kernel argument setters must bind the user-supplied argument index, not just append order.
- Copy byte counts must fit both source and destination buffers.
- NVRTC programs and loaded CUDA modules should be destroyed on every failure path after allocation.

## Potential Invariant Violations and Bugs

**Caveat on all CUDA entries in this pass: SOURCE-VERIFIED ONLY.** No hardware/CUDA toolchain access was available; every "fixed" classification below is based on reading `sarek-cuda/*.ml` and matching it against the commit log (`49da4768..618768b7`), not on running the code against a real device.

- **WITHDRAWN (verified 2026-07-02):** the prior entry claimed `sarek-cuda/Cuda_error.ml:22` (`let module_load_failed ptx_size reason = module_load_failed ptx_size reason`) is "self-recursive, likely bug." That is wrong — the definition has no `rec` keyword, so the right-hand-side `module_load_failed` resolves to the previously-bound name introduced by `include Sarek_backend_error.Backend_error.Make (...)` at line 14, not to itself. It is a valid, non-recursive delegation: a same-named rebinding kept only for backward compatibility (the parameter is called `ptx_size` here vs. whatever name the included functor uses). Correct classification: dead-weight rename wrapper (maintainability nit — an unnecessary shadowing shim), not a correctness bug.
- **fixed 2026-07-02 (merged, source-verified only):** `Device.destroy` previously left a stale entry in `device_cache`. It now evicts the entry and invokes registered destroy hooks (used to unload cached CUDA modules) before/while tearing down the context: `Hashtbl.remove device_cache dev.id` at `sarek-cuda/Cuda_api.ml:207`, driven by a `device_destroy_hooks` list (`Cuda_api.ml:49-53`) that `Kernel` registers into so `Kernel.cache` entries for the destroyed device are evicted too (commit `ef5d2775`, "evict Kernel.cache when destroying a CUDA context"); `cuModuleUnload` is called on eviction at `Cuda_api.ml:368` and `Cuda_api.ml:517`.
- Allocation computes `size * elem_size` without negative-size or overflow validation at `sarek-cuda/Cuda_api.ml:191-205`. Not addressed by this pass — still open.
- Copy APIs do not validate the opposite side capacity. `host_to_device`, `device_to_host`, and `device_to_device` use one side's byte count at `sarek-cuda/Cuda_api.ml:211-238`. **Still live** — no size-validation fix found in `49da4768..618768b7`; keep this open per the audit's explicit "keep live" instruction for copy-size gaps.
- **fixed 2026-07-02 (merged, source-verified only):** kernel cache keys previously omitted `name`. `compile_cached` (`sarek-cuda/Cuda_api.ml:498-500`) now builds its key with `Spoc_framework.Compile_cache.make_key ~device ~name ~source ()`, which digests device id, kernel/entry name, source, and options independently before joining — a source containing multiple kernels can no longer collide on the same cache entry.
- **fixed 2026-07-02 (merged, source-verified only):** `Cuda_plugin_base.Kernel.set_arg_*` previously ignored the supplied `idx` and appended to a list. It now stores each argument through the shared `Spoc_framework.Kernel_args` container (`type args = Cuda_api.Kernel.arg Spoc_framework.Kernel_args.t`, `Cuda_plugin_base.ml:136`), which does `Hashtbl.replace` keyed on `idx` (last-write-wins) and validates a contiguous `0..expected_count-1` index range before launch (`Cuda_plugin_base.ml:150-180`, `Kernel_args.validate_and_extract`).
- `Cuda_nvrtc.compile_to_ptx` destroys the program on compile failure and success, but if `nvrtcGetPTXSize` or `nvrtcGetPTX` fails after a successful compile, the program can leak at `sarek-cuda/Cuda_nvrtc.ml:379-387`. **Not addressed by this pass** — no evidence of a fix; keep this open (NVRTC program leak, per the audit's explicit "keep live" list).
- **KEEP LIVE — arity-validation gap, documented in code (verified 2026-07-02, source-verified only):** `Cuda_plugin_base.Kernel.launch` cannot validate true kernel arity because `Cuda_api.Kernel.t` carries no arity metadata. `expected_count` falls back to `Spoc_framework.Kernel_args.count args` (the number of distinct indices actually set), with an explicit code comment: `"KNOWN GAP: ... expected_count falls back to the number of distinct indices actually set. This still rejects internal gaps/duplicates but cannot catch a caller that consistently omits a trailing argument."` (`Cuda_plugin_base.ml:172-176`). So gaps/duplicates/out-of-range indices are caught by `Kernel_args.validate_and_extract`, but a caller who sets args `0..N-2` and simply never sets index `N-1` (pure trailing under-count) passes validation with `expected_count = N-1`. This is a real, intentional, code-documented gap — not a regression from the #213 fix, and not closed by it.
- `Sarek_ir_cuda.EVariant` emits `make_` plus the raw type name at `sarek/codegen/Sarek_ir_cuda.ml:159-160` (moved from the now-stub `sarek-cuda/Sarek_ir_cuda.ml`), while variant constructors are generated with the mangled type name at `sarek/codegen/Sarek_ir_cuda.ml:831`. Type names containing `.` likely produce mismatched or invalid CUDA.
- Nullary variant expressions emit the bare constructor tag at `sarek/codegen/Sarek_ir_cuda.ml:158`, which may not be a value of the enclosing variant struct. Marked likely bug pending typed IR examples.
- `atomic_add` supports two or three args but reports expected count `3` in the error path at `sarek/codegen/Sarek_ir_cuda.ml:318-336`.

## Performance and Maintainability Risks

- `initialized`, `device_cache`, and kernel cache refs are unsynchronized. Concurrent initialization, destroy, or compile can race.
- **fixed 2026-07-02 (merged, source-verified only):** `Cuda_api` previously raised a local exception separate from the structured `Cuda_error` module ("two different CUDA error surfaces"). `Cuda_api.check` now raises the canonical `Cuda_error.Cuda_error` (a `Backend_error` alias) via the shared `Sarek_backend_error.Backend_error.Make.check` funnel (`sarek-cuda/Cuda_api.ml:39-`); the old local `exception Cuda_error of cu_result * string` is kept only as an `[@ocaml.deprecated]` alias for opam-published out-of-tree callers (`Cuda_api.ml:25-31`). Note the payload changed too: the canonical path stringifies the `cu_result` via `to_string` before wrapping it, so catchers of the new exception get a formatted string, not the typed `cu_result` — see `kb/backends/shared-patterns.md` for the cross-backend payload note.
- README module sizes and API descriptions are stale. For example, `Cuda_error.ml` is documented as much larger than the current 22-line file, and several listed API names do not correspond to the current modules.
- The NVRTC architecture fallback clamps capabilities above 9.0 to `compute_90` at `sarek-cuda/Cuda_api.ml:318-327`; this is conservative but may leave newer hardware under-targeted until updated.

## Related Tests

- `sarek-cuda/test/dune:3-15` defines Alcotest suites with bisect instrumentation.
- `sarek-cuda/test/test_cuda_error.ml:118-137` covers shared error constructors and formatting.
- `sarek-cuda/test/test_sarek_ir_cuda.ml:253-277` covers literals, basic operations, statements, declarations, control flow, and helper pieces of code generation.

## Missing Tests

- Runtime test for `Device.destroy` followed by `Device.get` (behavior now fixed 2026-07-02, source-verified only; a runtime/hardware regression test is still missing).
- Kernel cache test with the same CUDA source containing two kernel names (`Compile_cache.make_key` fix is source-verified only; no runtime regression test found).
- Out-of-order and repeated `set_arg_*` tests (`Kernel_args` is unit-tested generically at `spoc/framework/test/test_kernel_args.ml`, but no CUDA-specific integration test was found).
- Allocation/copy overflow and bounds tests.
- NVRTC cleanup test for post-compile failure paths.
- Full generated CUDA C compile tests for records, variants, shared memory, atomics, and vector length parameters.

## Concrete Improvement Candidates

- ~~Include `name` in the CUDA kernel cache key and add a regression test with two kernels in one source.~~ **DONE 2026-07-02** — `Compile_cache.make_key` now includes `name` (source-verified only).
- ~~Store kernel args by index, validate contiguous required arguments before launch, and support replacement.~~ **DONE 2026-07-02** — via shared `Kernel_args` (source-verified only). Remaining gap: pure trailing under-count is still unvalidated (see arity note above; intentional, code-documented, not part of this fix).
- ~~Remove stale `device_cache` entries in `Device.destroy`, or make contexts process-lifetime and document that `destroy` is not supported.~~ **DONE 2026-07-02** — `Device.destroy` now evicts the cache entry and runs destroy hooks that unload cached CUDA modules (source-verified only).
- Wrap NVRTC program lifetime in `Fun.protect` after creation. **Still open** — no evidence of a fix in this pass.
- Add byte-size validation helpers shared by `Memory.host_to_device`, `device_to_host`, and `device_to_device`. **Still open** — copy-size validation gap explicitly kept live per this audit's instructions.
- Fix variant constructor mangling and add a generated-code test for namespaced variants. **Still open** — not covered by this pass.
