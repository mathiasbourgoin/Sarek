# CPU Runtime

## Component Inventory

Reviewed CPU runtime files: the `Sarek_cpu_runtime*` module family, `sarek/sarek/Sarek_cpu_runtime.mli`, `sarek/sarek/BSP.md`, and runtime tests in `sarek/sarek/test/**` that touch related error/value helpers. Interpreter-local pools are covered in [interpreter.md](interpreter.md). The former monolithic `Sarek_cpu_runtime.ml` (~1497 lines) was split (pure move) into four modules; the `.mli` is unchanged.

## Per-File Purpose

- `sarek/sarek/Sarek_cpu_runtime_types.ml`: `exec_mode`, `thread_state`, shared-memory types/allocators, and the `Barrier` effect (`sarek/sarek/Sarek_cpu_runtime_types.ml:10-99`).
- `sarek/sarek/Sarek_cpu_runtime_exec.ml`: barrier-aware sequential execution — `run_block_sequential_bsp` (`sarek/sarek/Sarek_cpu_runtime_exec.ml:17`), `run_sequential` (`:132`), and `run_block_with_barriers` (`:166`).
- `sarek/sarek/Sarek_cpu_runtime_pools.ml`: the pool implementations — `DomainPool` (`sarek/sarek/Sarek_cpu_runtime_pools.ml:9`), `ThreadPool` (`:126`), `ParallelPool` (`:525`), and `LaunchQueue` (`:691`).
- `sarek/sarek/Sarek_cpu_runtime.ml` (reduced): the public `run`/`run_parallel_simple` orchestrators that select barrier-aware or non-barrier execution and re-export the moved entry points (`sarek/sarek/Sarek_cpu_runtime.ml:66-78`).
- `sarek/sarek/Sarek_cpu_runtime.mli`: public CPU runtime interface (unchanged by the split).
- `sarek/sarek/BSP.md`: barrier and BSP design intent.

## Features/APIs

- CPU fallback execution of kernels over grid/block dimensions.
- Barrier-aware sequential execution using effect-like exceptions.
- Domain pool and thread pool implementations for parallel block execution.
- Specialized paths for fission and parallel execution strategies.
- Shared memory and parameter environment setup for CPU kernel calls.
- Public run helpers that select barrier-aware or non-barrier execution.

## Invariants

- Each logical work-item should execute exactly once unless the kernel explicitly loops.
- Barrier detection must not mutate user-visible state.
- A block with barriers must only progress when all live work-items have reached a compatible barrier or completed validly.
- Worker exceptions must propagate to the caller and must not leave pending counters unreleased.
- Grid and block dimensions must be positive and small enough to avoid overflow in product calculations.
- Global pools must have a lifecycle that does not leak domains/threads indefinitely.

## Potential Invariant Violations/Bugs

- Barrier divergence can be accepted. `run_block_with_barriers` loops over waiting/completed threads without a convergence/deadlock check at `sarek/sarek/Sarek_cpu_runtime_exec.ml:267-279` (function at `sarek/sarek/Sarek_cpu_runtime_exec.ml:166`); `ThreadPool.run_block_bsp` has the same issue at `sarek/sarek/Sarek_cpu_runtime_pools.ml:253` (function at `sarek/sarek/Sarek_cpu_runtime_pools.ml:169`). The sequential deep handler has a check at `sarek/sarek/Sarek_cpu_runtime_exec.ml:113-127` (in `run_block_sequential_bsp`), but it does not cover the main pool paths.
- Other pool paths can deadlock on exceptions because pending counters are decremented after work completes. `ThreadPool.worker_fn` runs tasks at `sarek/sarek/Sarek_cpu_runtime_pools.ml:323-367` (decrement at `:366-367`), and `ParallelPool.worker_fn` does so at `sarek/sarek/Sarek_cpu_runtime_pools.ml:550-586` (decrement at `:585-586`); an exception before decrement/signaling can leave waiters blocked.
- Dimension products are computed without clear positive-dimension/overflow validation, for example at `sarek/sarek/Sarek_cpu_runtime.ml:77-79`, `sarek/sarek/Sarek_cpu_runtime.ml:199-201`, `sarek/sarek/Sarek_cpu_runtime_exec.ml:20`, `sarek/sarek/Sarek_cpu_runtime_exec.ml:169`, and `sarek/sarek/Sarek_cpu_runtime_pools.ml:176`, `sarek/sarek/Sarek_cpu_runtime_pools.ml:407-409`.

## Recently Resolved

- Side-effecting barrier detection was removed by PR #137, merged as `d30b2ba`. Callers can now provide explicit `has_barriers` metadata, and omitted metadata takes a conservative sequential barrier-capable path instead of probing the user kernel.
- The reviewed DomainPool path now records and re-raises worker failures instead of swallowing them; this was covered by PR #137.

## Performance Or Maintainability Risks

- There are several independent execution strategies now spread across the `Sarek_cpu_runtime_exec`/`Sarek_cpu_runtime_pools`/`Sarek_cpu_runtime` modules, making semantic drift likely.
- Global pools are created lazily at `sarek/sarek/Sarek_cpu_runtime_pools.ml:103-111` (DomainPool `global_pool`), `sarek/sarek/Sarek_cpu_runtime_pools.ml:510` (fission pool), and `sarek/sarek/Sarek_cpu_runtime_pools.ml:675` (parallel pool); shutdown helpers exist but are not obviously wired into runtime lifecycle.
- Barrier support depends partly on runtime detection and partly on metadata-like flags in later paths, which makes behavior input-path dependent.
- Exception handling differs by pool implementation, making failures hard to reproduce.
- Shared-memory and local-environment setup is duplicated across execution paths.

## Related Tests

The colocated `sarek/sarek/test/**` files cover supporting value/error/helper modules. Focused CPU runtime regressions now live under `sarek/tests/unit/test_cpu_runtime.ml`, including non-mutating barrier selection and default-path exception propagation from PR #137.

## Missing Tests

- Divergent barrier kernel where only some work-items reach a barrier, expecting a structured deadlock/error.
- Worker exception propagation for `ThreadPool` and `ParallelPool`.
- Zero, negative, and very large grid/block dimensions.
- Shared-memory initialization and isolation across blocks.
- Global pool shutdown or non-leaking lifecycle behavior.

## Concrete Improvement/Fix Candidates

- Add a block-level barrier convergence check: all non-completed work-items must be waiting at the same barrier generation before resuming.
- Wrap worker task execution in `Fun.protect` so pending counters and condition variables are updated even when work raises.
- Capture the first worker exception and re-raise it in the caller after the pool drains.
- Validate grid and block dimensions at public entrypoints before computing products.
- Centralize common environment/shared-memory setup to reduce path-specific semantic drift.
