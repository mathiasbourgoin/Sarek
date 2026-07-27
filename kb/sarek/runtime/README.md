# Sarek Runtime Knowledge Base

Scope reviewed: `sarek/README.md`, `sarek/core/**`, `sarek/framework/**`, `sarek/sarek/**`, `sarek/interp/**`, `sarek/execute/**`, `sarek/Sarek_stdlib/**`, `sarek/Sarek_float64/**`, `sarek/Sarek_geometry/**`, `sarek/Visibility_lib/**`, `sarek/plugins/native/**`, `sarek/plugins/interpreter/**`, plus tests colocated in those directories. Excluded for this pass: `sarek/ppx/**`, `sarek/ppx_intrinsic/**`, GPU backends, and top-level `sarek/tests/**` except where runtime docs referenced them.

**2026-07-02 directory-move note:** the interpreter and Execute modules were pulled out of `sarek/sarek/` into their own dune libraries — `sarek/interp/` (`sarek_interp`) and `sarek/execute/` (`sarek_execute`). `sarek/sarek/` (library `sarek`) keeps 6-line forwarding shims for the public `Sarek.*` namespace. See [interpreter.md](interpreter.md) and [execution.md](execution.md) for per-file detail. **Cleanup item resolved 2026-07-02 (merged, commit 7ca31a76):** the byte-identical dead copies of `Sarek_ir_interp_{value,intrinsics,eval}.ml` and `Skeletons.ml` that used to sit excluded from `sarek/sarek/dune`'s modules list have been deleted outright.

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
- `sarek/interp/Sarek_ir_interp*.ml` (library `sarek_interp`): interpreter semantics and direct execution. Split into `Sarek_ir_interp_value` (value/env/thread state, conversions), `Sarek_ir_interp_intrinsics` (gpu/float/int/type-conversion intrinsics), `Sarek_ir_interp_eval` (recursive eval/exec chain), and the reduced `Sarek_ir_interp` (DomainPool, run_grid*, run_kernel* API). `sarek/sarek/Sarek_ir_interp*.ml` are now 6-line forwarding shims; `sarek/sarek/Sarek_ir_interp_{value,intrinsics,eval}.ml` (plural files sharing the basenames) are separate, dead, byte-identical copies excluded from `sarek/sarek/dune`'s modules list (2026-07-02 audit finding).
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
- Interpreter `LArrayElem` (plain-name lvalue) writes still lack the explicit bounds check used elsewhere: `sarek/interp/Sarek_ir_interp_eval.ml:353-357`. **Fixed as of 2026-07-02:** `EArrayReadExpr` reads (`sarek/interp/Sarek_ir_interp_eval.ml:63-76`) and `LArrayElemExpr` writes (`:359-373`) are now bounds-checked — see [interpreter.md](interpreter.md) for detail.
- Cache `get`/`put` accept arbitrary key strings as path components: `sarek/framework/Framework_cache.ml:96-100` and `sarek/framework/Framework_cache.ml:124-131`. Still live as of 2026-07-02 — not the same module as the new `spoc/framework/Compile_cache.ml` (compile-cache *key shape*, unrelated to `Framework_cache`'s path-traversal risk).
- Interpreter int64 shift/bitwise ops still truncate to 32-bit: `Shl`/`Shr`/`BitAnd`/`BitOr`/`BitXor`/`BitNot` all route through `Int32` regardless of `VInt64` operands (`sarek/interp/Sarek_ir_interp_value.ml:245-256,267`). Still live as of 2026-07-02 — see [interpreter.md](interpreter.md).

## Recently Resolved

- Cross-device transfer from authoritative device data and authoritative-buffer cleanup were fixed by PR #136, merged as `5dffea3`.
- Side-effecting CPU barrier detection was removed by PR #137, merged as `d30b2ba`. The reviewed DomainPool path also now reports worker failures instead of silently succeeding.
- Fusion pipeline dropping of unfused kernels was fixed by PR #138, merged as `06b7d70`.
- **2026-07-02:** `ThreadPool` worker exception propagation fixed (try/with + `first_error`, `Sarek_cpu_runtime_pools.ml:367-398` — see [cpu-runtime.md](cpu-runtime.md)); `EArrayReadExpr`/`LArrayElemExpr` interpreter bounds checks added (see [interpreter.md](interpreter.md)); `Execute` custom-element placeholder replaced with a real byte round-trip (see [execution.md](execution.md)); interpreter legacy-launch byte-width type sniffing replaced by typed `exec_arg`s (see [plugins.md](plugins.md)). **Still open:** `ParallelPool` has the same exception-swallowing hazard `ThreadPool` used to have — not yet fixed.
- **2026-07-02 (merged, post-audit):** native and interpreter plugin `set_arg_*` functions no longer ignore the requested index — both now route through the shared `Spoc_framework.Kernel_args` indexed container (`spoc/framework/Kernel_args.ml`/`.mli`), which stores by `idx` (last-set-wins) and validates via `validate_and_extract` before launch (`sarek/plugins/native/Native_plugin_base.ml:565-757`, `sarek/plugins/interpreter/Interpreter_plugin_base.ml:528-680` — see [plugins.md](plugins.md)). Fusion atomic detection is real: `kernel_uses_atomics` (`spoc/ir/Sarek_ir_analysis.ml:209`) is the single source of truth and all four fusion gates (`can_fuse`, `can_fuse_reduction`, `can_fuse_stencil`, `should_fuse`) reject atomics (commit b4c78853 — see [fusion.md](fusion.md)). Dead byte-identical interpreter copies (`Sarek_ir_interp_value.ml`, `_intrinsics.ml`, `_eval.ml`, `Skeletons.ml` under `sarek/sarek/`) were deleted (commit 7ca31a76 — see [interpreter.md](interpreter.md)). Interpreter `Shr` is now `Int32.shift_right` (arithmetic), matching codegen (see [interpreter.md](interpreter.md)).

## Performance Or Maintainability Risks

- Several global registries and plugin registries are mutable `Hashtbl`s without synchronization.
- CPU pools are global and have shutdown functions that are not wired into lifecycle management.
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
- Fusion full-expression substitution (atomic rejection is now covered by `test_can_fuse_with_direct_atomic`/`test_can_fuse_with_atomic_in_helper`, `sarek/tests/unit/test_fusion.ml:173-245`).
- Plugin indexed-argument order (verify explicit out-of-order/overwrite-by-index plugin-level coverage beyond `Kernel_args`'s own unit tests) and legacy vector type handling.
- Framework cache path traversal, atomic writes, and concurrent registry/cache use.
- Interpreter int64 shift/bitwise truncation (still-live gap, see above).

## Concrete Improvement/Fix Candidates

- Add worker exception propagation with `Fun.protect` around pending counters and wait signaling.
- Validate cache keys or keep `Framework_cache.compute_key` internal to `get`/`put`.
- **Done 2026-07-02 (merged):** ~~enforce indexed plugin argument setting by replacing or resizing an argument array rather than prepending~~ — implemented via `Spoc_framework.Kernel_args`.
- Add focused regression tests before broad refactors, because these bugs sit at runtime boundary points.
- Give the interpreter's `Shl`/`BitAnd`/`BitOr`/`BitXor`/`BitNot` (and `Shr`'s already-fixed sibling ops) `VInt64`-preserving branches so 64-bit shift/bitwise operations stop truncating to 32-bit.
