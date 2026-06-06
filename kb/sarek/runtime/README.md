# Sarek Runtime Knowledge Base

Scope reviewed: `sarek/README.md`, `sarek/core/**`, `sarek/framework/**`, `sarek/sarek/**`, `sarek/Sarek_stdlib/**`, `sarek/Sarek_float64/**`, `sarek/Sarek_geometry/**`, `sarek/Visibility_lib/**`, `sarek/plugins/native/**`, `sarek/plugins/interpreter/**`, plus tests colocated in those directories. Excluded for this pass: `sarek/ppx/**`, `sarek/ppx_intrinsic/**`, GPU backends, and top-level `sarek/tests/**` except where runtime docs referenced them.

No source files were modified. This KB only covers runtime/core/framework support.

## Component Inventory

- [core.md](core.md): device abstraction, vectors, transfer state, memory accounting, kernels, runtime helpers, logging, profiling, and advanced placeholders.
- [framework.md](framework.md): backend framework registry, intrinsic registry, cache, and framework errors.
- [execution.md](execution.md): KIRC/Sarek IR types, kernel packaging, execution dispatch, skeleton helpers, errors, values, and type helpers.
- [cpu-runtime.md](cpu-runtime.md): CPU runtime, BSP/barrier execution, domain/thread pools, and CPU kernel launch paths.
- [interpreter.md](interpreter.md): IR interpreter, direct execution path, interpreter-local parallel helpers, and interpreter errors.
- [fusion.md](fusion.md): fusion analysis, map/map and stencil fusion rewrites, cost model, and fusion diagnostics.
- [stdlib-and-support.md](stdlib-and-support.md): standard library intrinsics, float32/float64 support, geometry, and visibility support packages.
- [plugins.md](plugins.md): native and interpreter plugin registration, memory/kernel shims, direct execution hooks, and plugin tests.

## Per-File Purpose

Per-file purpose is documented in each subcomponent page. The highest-risk files from this pass are:

- `sarek/core/Transfer.ml`: vector host/device transfer state and stale-location transitions.
- `sarek/sarek/Sarek_cpu_runtime*.ml`: CPU execution strategy, barriers, pools, and fallback execution. Split into `Sarek_cpu_runtime_types` (exec/thread/shared-mem types, Barrier effect), `Sarek_cpu_runtime_exec` (sequential/BSP/barrier execution), `Sarek_cpu_runtime_pools` (DomainPool/ThreadPool/ParallelPool/LaunchQueue), and the reduced `Sarek_cpu_runtime` (public run orchestrators).
- `sarek/sarek/Sarek_ir_interp*.ml`: interpreter semantics and direct execution. Split into `Sarek_ir_interp_value` (value/env/thread state, conversions), `Sarek_ir_interp_intrinsics` (gpu/float/int/type-conversion intrinsics), `Sarek_ir_interp_eval` (recursive eval/exec chain), and the reduced `Sarek_ir_interp` (DomainPool, run_grid*, run_kernel* API).
- `sarek/sarek/Sarek_fusion.ml`: fusion eligibility and rewrite implementation.
- `sarek/framework/Framework_cache.ml`: cache paths and serialized artifacts.
- `sarek/plugins/native/Native_plugin_base.ml` and `sarek/plugins/interpreter/Interpreter_plugin_base.ml`: plugin-facing memory/kernel compatibility layers.

## Features/APIs

The runtime slice exposes:

- A typed vector abstraction with host/device location metadata.
- Abstract devices, memory info, transfers, and kernel launch primitives.
- Framework and intrinsic registries for backend integration.
- KIRC/Sarek IR execution through native backends, interpreter backends, and CPU fallback.
- CPU BSP/barrier execution helpers and parallel pool strategies.
- Fusion passes for map pipelines and stencil-like kernels.
- Standard math/GPU intrinsics and float32/float64 support modules.
- Native and interpreter plugins that register runtime backends.

## Invariants

- Vector location metadata must identify the authoritative copy of data before any transfer, free, or gather operation.
- Kernel argument order and type must remain stable from user-facing APIs through backend launch.
- CPU BSP execution must not run side-effecting kernel code more times than the logical grid requires.
- Barrier semantics require all live work-items in a block to reach compatible barriers before progress.
- Interpreter and CPU runtimes must propagate worker exceptions to callers instead of reporting success or hanging.
- Framework cache keys must map to files inside the cache directory only.
- Fusion must preserve all kernels in a pipeline unless it proves and performs a semantics-preserving replacement.
- Plugin registries and global runtime registries should be safe under repeated registration and concurrent use, or explicitly documented as single-threaded initialization-only APIs.

## Potential Invariant Violations/Bugs

- CPU/thread-pool barrier loops can resume divergent barriers instead of reporting deadlock: `sarek/sarek/Sarek_cpu_runtime_exec.ml:267-279` (in `run_block_with_barriers`) and `sarek/sarek/Sarek_cpu_runtime_pools.ml:253` (in `ThreadPool.run_block_bsp`).
- Interpreter array expression reads and writes lack the explicit bounds checks used by simple array reads: `sarek/sarek/Sarek_ir_interp_eval.ml:63-72` and `sarek/sarek/Sarek_ir_interp_eval.ml:350-363`.
- Cache `get`/`put` accept arbitrary key strings as path components: `sarek/framework/Framework_cache.ml:96-100` and `sarek/framework/Framework_cache.ml:124-131`.
- Native and interpreter plugin `set_arg_*` functions ignore the requested index and rely on call order: `sarek/plugins/native/Native_plugin_base.ml:568-617` and `sarek/plugins/interpreter/Interpreter_plugin_base.ml:536-570`.

## Recently Resolved

- Cross-device transfer from authoritative device data and authoritative-buffer cleanup were fixed by PR #136, merged as `5dffea3`.
- Side-effecting CPU barrier detection was removed by PR #137, merged as `d30b2ba`. The reviewed DomainPool path also now reports worker failures instead of silently succeeding.
- Fusion pipeline dropping of unfused kernels was fixed by PR #138, merged as `06b7d70`.

## Performance Or Maintainability Risks

- Several global registries and plugin registries are mutable `Hashtbl`s without synchronization.
- CPU pools are global and have shutdown functions that are not wired into lifecycle management.
- Fusion analysis currently records atomics as always absent, which makes the safety model hard to trust.
- Multiple compatibility paths coexist: direct execution, legacy `Kernel.launch`, CPU fallback, and interpreter plugin launch. Their transfer and argument semantics differ.
- Some advanced APIs appear placeholder-like but are public enough to create support debt.

## Related Tests

- Core colocated tests cover device construction, vectors, vector storage, vector transfer, memory, kernels, kernel args, and GPU memory.
- Framework tests cover framework registry behavior, intrinsic registry behavior, cache basics, and dummy-backend integration.
- Sarek tests cover values, float32 helpers, type helpers, and error modules.
- Native and interpreter plugin tests currently cover error modules only.

## Missing Tests

- Overlapping vector blits and subvector gather/partition semantics.
- CPU divergent barriers, remaining pool exception propagation paths, and invalid dimension handling.
- Interpreter array bounds, parameter/argument mismatch, variant tag collisions, and writeback behavior.
- Fusion atomic rejection and full-expression substitution.
- Plugin indexed-argument order and legacy vector type handling.
- Framework cache path traversal, atomic writes, and concurrent registry/cache use.

## Concrete Improvement/Fix Candidates

- Add worker exception propagation with `Fun.protect` around pending counters and wait signaling.
- Validate cache keys or keep `Framework_cache.compute_key` internal to `get`/`put`.
- Enforce indexed plugin argument setting by replacing or resizing an argument array rather than prepending.
- Add focused regression tests before broad refactors, because these bugs sit at runtime boundary points.
