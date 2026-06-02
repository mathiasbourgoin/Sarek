# Intake + Plan — ir-interp-split

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

`sarek/sarek/Sarek_ir_interp.ml` (1573 lines) splits by responsibility. PURE MOVE,
byte-identical behavior. NO `.mli` exists, so the whole module is public API — every
externally-referenced name must remain reachable as `Sarek_ir_interp.<name>`.

## External API that MUST stay under `Sarek_ir_interp` (verified by grep)

Used by `Execute.ml`, `sarek/plugins/interpreter/*`, and e2e tests:
- type `value` **and its constructors** (`VInt32`, `VInt64`, `VFloat32`, `VFloat64`,
  `VRecord`, `VUnit`, …) — note `value = Sarek_value.value`
- type `arg` (`ArgArray`, `ArgScalar`)
- type `writeback` (`Writeback`)
- `to_int`, `to_float32`/`to_float64` (whatever the digit-suffixed names actually are)
- `parallel_mode`, `run_kernel`, `run_kernel_with_exec_args`, `exec_vector_to_array`

Any of these that move to a sub-module MUST be re-exported from the main module with
type equality (constructors copied verbatim for `value`), exactly like the
`thread_state` re-export in the cpu-runtime split (Intake 4).

## Module DAG

```
value  ←  intrinsics  ←  eval  ←  Sarek_ir_interp (main)
```

1. **`Sarek_ir_interp_value.ml`** (leaf) — lines ~17–314:
   `module F32`, `type value = Sarek_value.value = <ctors>`, `Barrier` effect,
   `thread_state`, `env`, `create_env`, `copy_env`, `bind_var`, `lookup_var`,
   `to_int32/int64/int/float32/float64/bool`, `eval_binop`, `eval_unop`, path predicates
   (`is_gpu_path`, `is_float32_path`, `is_float64_path`, `is_int32_path`).
2. **`Sarek_ir_interp_intrinsics.ml`** — lines ~315–609: `eval_gpu_index_intrinsic`,
   `eval_barrier_intrinsic`, `eval_float32_math_intrinsic`, `eval_float64_math_intrinsic`,
   `eval_int32_math_intrinsic`, `eval_type_conversion_intrinsic`. `open …_value`.
3. **`Sarek_ir_interp_eval.ml`** — lines ~610–1115: the `let rec eval_intrinsic … and …`
   chain through `exec_stmt_for_return`, plus `run_block`, `run_grid_sequential`. This
   recursive chain is ONE unit — keep it intact. `open …_value`, `open …_intrinsics`.
4. **`Sarek_ir_interp.ml`** (main) — lines ~1116–end: `DomainPool`, `global_pool`,
   `get_pool`, `run_grid_parallel`, `parallel_mode`, `run_grid`, type `arg`, `run_kernel`,
   vector↔array conversions, `writeback`/`exec_writeback`, exec-args machinery,
   `run_kernel_with_exec_args`, `run_kernel_with_args`. `open` the three sub-modules.
   Plus RE-EXPORTS of the external-API names listed above (value+ctors, to_int,
   to_float*, exec_vector_to_array).

## Quality Gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek/sarek/
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
ocamlformat --check sarek/sarek/Sarek_ir_interp*.ml
```

`@sarek/tests/runtest` builds `Execute.ml`, the interpreter plugin, and
`test_polymorphism.ml` — the external callers — so it is the real gate for the
re-exports. Ignore `-lnvrtc` from a full build.

## Risks

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| External caller loses access to a moved name | Med | Med | re-export all listed names; runtest builds the callers |
| `value` constructor visibility lost (abstract re-export) | Med | High | re-export `type value = …_value.value = <ctors>`; runtest gates |
| rec eval chain split mid-way | Low | High | keep 610–1115 as one unit in eval module |
| Logic changed during move | Low | High | pure move; runtest |
