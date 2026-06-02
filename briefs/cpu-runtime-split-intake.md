# Intake + Plan — cpu-runtime-split

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

`sarek/sarek/Sarek_cpu_runtime.ml` (1497 lines) splits by responsibility. PURE MOVE,
byte-identical behavior. The module has a public `.mli` (215 lines) — the public surface
must remain identical.

## Scope Boundary

OUT of scope (move verbatim, do NOT fix — each is a separate future intake):
- Barrier-divergence deadlock (no convergence check in `run_block_with_barriers`,
  `ThreadPool.run_block_bsp`).
- Exception-before-decrement deadlock in `ThreadPool.worker_fn` / `ParallelPool.worker_fn`.
- Missing grid/block dimension validation / overflow checks.
No logic changes. The `.mli` public interface is unchanged.

## Module DAG (verified by code reading)

```
types  ←  exec  ←  pools  ←  Sarek_cpu_runtime (main, keeps .mli)
   ↖________________________/
```

1. **`Sarek_cpu_runtime_types.ml`** (leaf) — current lines ~13–154:
   `module Float32 = Sarek_float32`, `exec_mode`, `thread_state`, `global_idx_*`,
   `global_size_*`, `any_array`, `shared_mem`, `create_shared`, `alloc_shared_*`,
   and `type _ Effect.t += Barrier`.
2. **`Sarek_cpu_runtime_exec.ml`** — lines ~156–420:
   `run_block_sequential_bsp`, `run_sequential`, `run_block_with_barriers`. Depends on
   types. (Verified: references no pool.)
3. **`Sarek_cpu_runtime_pools.ml`** — the internal pool machinery:
   `DomainPool` (422–515), `ThreadPool` (644–1015), `ParallelPool` (1043–1177),
   `LaunchQueue` (1286–1392), plus their global-ref/getter helpers
   (`global_pool`/`get_pool`, `fission_pool`/`get_fission_pool`,
   `parallel_pool`/`get_parallel_pool`, `fission_queues`/`get_fission_queue`).
   Depends on types (+ exec if a pool body calls run_block_*; the implementer confirms
   by compiler).
4. **`Sarek_cpu_runtime.ml`** (main, KEEPS the existing `.mli`) — the public orchestrators:
   `run_parallel_simple`, `run_parallel_with_barriers`, `run_parallel`, `run_threadpool`,
   `enqueue_fission`, `flush_fission_queue`, `flush_fission`, `run_1d/2d/3d_threadpool`.
   Re-exports the public types/helpers from the types module (see below).

## .mli re-export requirement (the careful part)

The `.mli` exposes `exec_mode`, `thread_state` (concrete records/variants), `shared_mem`
(abstract), and the `global_*` / `create_shared` / `alloc_shared_*` helpers. After moving
their definitions to `Sarek_cpu_runtime_types`, the main `.ml` must re-export them with
type equality so the `.mli` still matches and external callers see identical types:

```ocaml
type exec_mode = Sarek_cpu_runtime_types.exec_mode = <constructors copied verbatim>
type thread_state = Sarek_cpu_runtime_types.thread_state = { <fields copied verbatim> }
type shared_mem = Sarek_cpu_runtime_types.shared_mem   (* abstract in .mli — plain alias ok *)
let global_idx_x = Sarek_cpu_runtime_types.global_idx_x
(* … all other public helpers re-exported … *)
```

The `.mli` file itself is NOT changed.

## Quality Gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek/sarek/
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
ocamlformat --check sarek/sarek/Sarek_cpu_runtime*.ml
```

(`@sarek/tests/runtest` includes `test_cpu_runtime.ml` — the real regression gate.
Ignore `-lnvrtc` from a full build.)

## Risks

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| .mli re-export type mismatch | Med | Low (compile error) | incremental build; copy constructors/fields verbatim |
| Concurrency logic altered during move | Low | High | pure move, no edits inside bodies; test_cpu_runtime gates |
| Accidentally fixing a documented deadlock/overflow bug | Low | High | forbidden; reviewer checks the flagged sites are verbatim |
| Pool→exec dependency missed | Med | Low (compile error) | DAG allows pools to open exec; compiler confirms |
