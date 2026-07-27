# Interpreter Runtime

## Component Inventory

Reviewed interpreter runtime files: the `Sarek_ir_interp*` module family (now the `sarek_interp` library under `sarek/interp/`), `sarek/plugins/interpreter/**`. Plugin-specific notes are also summarized in [plugins.md](plugins.md). The former monolithic `Sarek_ir_interp.ml` (~1573 lines) was split (pure move) into four modules.

**Directory move (2026-07-02 audit):** the interpreter now lives entirely under `sarek/interp/` as its own dune library `sarek_interp` (`public_name sarek.interp`, `sarek/interp/dune`). `sarek/sarek/` (library `sarek`) no longer holds the live implementation — it keeps 6-line forwarding shims (`include Sarek_interp.<Module>`) for the public `Sarek.*` namespace, e.g. `sarek/sarek/Sarek_ir_interp.ml` and `sarek/sarek/Interp_error.ml` are each 6 lines.

**Dead byte-identical copies — removed 2026-07-02 (merged, commit 7ca31a76, "delete dead byte-identical interpreter copies and v1 relic").** `sarek/sarek/Sarek_ir_interp_value.ml`, `Sarek_ir_interp_intrinsics.ml`, `Sarek_ir_interp_eval.ml`, and `Skeletons.ml` (the byte-identical duplicates of the real `sarek/interp/` sources, ~1,124 dead lines) are gone. Verified 2026-07-02: `find sarek/sarek -maxdepth 1 -iname '*interp*'` now returns only the legitimate 6-line forwarding shims `sarek/sarek/Sarek_ir_interp.ml` and `sarek/sarek/Interp_error.ml` described below — no `Sarek_ir_interp_value.ml`/`_intrinsics.ml`/`_eval.ml`/`Skeletons.ml` remain in that directory.

## Per-File Purpose

- `sarek/interp/Sarek_ir_interp_value.ml`: the `value` type, environment, thread state, `to_*` conversions, `eval_binop`/`eval_unop`, and path predicates (`sarek/interp/Sarek_ir_interp_value.ml:10-280`).
- `sarek/interp/Sarek_ir_interp_intrinsics.ml`: GPU/float/int/type-conversion intrinsics, split out of the former monolithic `eval_intrinsic` (`sarek/interp/Sarek_ir_interp_intrinsics.ml:14-323`).
- `sarek/interp/Sarek_ir_interp_eval.ml`: the recursive `eval_expr`/exec chain over expressions, statements, and lvalues (`sarek/interp/Sarek_ir_interp_eval.ml`).
- `sarek/interp/Sarek_ir_interp.ml` (reduced): interpreter-local `DomainPool`, the `run_grid*` drivers, and the public `run_kernel*` API (`sarek/interp/Sarek_ir_interp.ml:78-535`).
- `sarek/interp/Interp_error.ml`: interpreter error variants and formatting.
- `sarek/sarek/Sarek_ir_interp.ml`, `sarek/sarek/Interp_error.ml`: 6-line forwarding shims (`include Sarek_interp.<Module>`) preserving the `Sarek.*` namespace. (The byte-identical dead copies that used to share these basenames plus `_value`/`_intrinsics`/`_eval`/`Skeletons` suffixes were deleted 2026-07-02 — see above.)
- `sarek/plugins/interpreter/Interpreter_plugin.ml`: framework plugin registration for the interpreter backend.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml`: backend-compatible memory/kernel facade and direct execution integration.
- `sarek/plugins/interpreter/Interpreter_error.ml`: plugin error variants.
- `sarek/plugins/interpreter/test/test_interpreter_error.ml`: plugin error tests.

## Features/APIs

- Expression and statement interpretation over boxed runtime values.
- Array, tuple, record, option, result, and variant handling.
- Kernel argument conversion from core `Kernel_arg` into interpreter values.
- Direct execution path used by the interpreter plugin.
- Parallel/domain helper code for interpreter execution.
- Structured interpreter error reporting for selected failure modes.

## Invariants

- All array and vector accesses should perform consistent bounds checks and report interpreter errors.
- Kernel parameter binding should consume only user parameters and should reject count/type mismatches clearly.
- Variant tags should be stable and collision-free for pattern matching.
- Interpreter worker exceptions must propagate to the caller.
- Direct execution and legacy plugin execution should agree on vector element types and writeback semantics.

## Potential Invariant Violations/Bugs

- **Fixed as of 2026-07-02** (was: inconsistent bounds checks): `EArrayReadExpr` now bounds-checks and raises `Array_bounds_error` (`sarek/interp/Sarek_ir_interp_eval.ml:63-76`), matching `EArrayRead` (`:55-62`). `LArrayElemExpr` assignment is also now bounds-checked (`sarek/interp/Sarek_ir_interp_eval.ml:359-373`). **Still unchecked:** `LArrayElem` write (plain-name lvalue) indexes directly with `a.(i) <- value` and has no bounds check — `sarek/interp/Sarek_ir_interp_eval.ml:353-357`.
- `run_kernel` binds declarations and arguments with `List.iter2` at `sarek/interp/Sarek_ir_interp.ml:202-208`. If declarations include non-parameter entries or the argument count differs, callers can get a generic `Invalid_argument` or wrong binding. The later `args_from_kernel_args` path at `sarek/interp/Sarek_ir_interp.ml:469-528` is more explicit.
- Variant tags are computed from `Hashtbl.hash ctor mod 256` when constructing and matching variants at `sarek/interp/Sarek_ir_interp_eval.ml:126-127`, `sarek/interp/Sarek_ir_interp_eval.ml:151`, and `sarek/interp/Sarek_ir_interp_eval.ml:286`. Constructor collisions are possible.
- **Fixed 2026-07-03 (merged, PR #217).** Interpreter-local `DomainPool.worker` no longer discards task exceptions: the first error (with raw backtrace) is captured on `pool.first_error` and re-raised from `wait_all` via `Printexc.raise_with_backtrace` — the "interpreter worker exceptions must propagate" invariant now holds. The same PR fixed the `DomainPool.create` record-aliasing bug (`{pool with domains}` split `active_tasks` into two cells, letting `wait_all` return while a block still ran — root cause of CI's intermittent `dst[768]=0` drop at 96 cores), made `get_pool` init `Fun.protect`-safe, and added the `SAREK_DOMAIN_COUNT` override + `test_domain_pool_stress.ml` regression test.

## Recently Resolved

- **`Shr` canonicalized to arithmetic (sign-extending) shift — fixed 2026-07-02 (merged).** `eval_binop`'s `Shr` case now uses `Int32.shift_right` (arithmetic) rather than a logical shift, matching every codegen backend: CUDA/OpenCL/Metal/GLSL/WGSL emit plain `>>` on a signed int type and PTX emits `shr.s32` (`sarek/interp/Sarek_ir_interp_value.ml:246-253`). Logical shift (`lsr`) is intentionally lowered to a separate expression tree in `Sarek_lower_ir.ml` precisely because the `Shr` IR node is arithmetic — so the interpreter's `Shr` semantics now agree with codegen instead of silently diverging on negative operands.

**Still live (unfixed, pre-existing) — do not weaken:** `Shl`/`Shr`/`BitAnd`/`BitOr`/`BitXor`/`BitNot` in `eval_binop`/`eval_unop` all route through `to_int32`/produce `VInt32`, regardless of whether an operand is `VInt64` (`sarek/interp/Sarek_ir_interp_value.ml:245-256,267`). There is no `VInt64`-preserving branch for any shift or bitwise op (contrast `Add`/`Sub`/`Mul`/`Div`/`Rem`/`Neg`, which do have explicit `VInt64` branches, `:172-211,264`). A 64-bit shift/bitwise operation is silently truncated to 32-bit through this path. Verified still present 2026-07-02; genuinely unrelated to the `Shr` sign-extension fix above, which only changed which C-level shift the `Int32` case performs, not whether `Int64` operands get their own path.

## Performance Or Maintainability Risks

- Direct interpreter execution and plugin legacy `Kernel.launch` have different argument paths, increasing semantic drift.
- The evaluator has many expression cases with manual recursion; new expression variants can miss checks or substitution-like logic.
- Hash-derived variant tags make debugging and cross-run compatibility harder than explicit constructor IDs.
- Interpreter-local parallel helpers duplicate CPU runtime pool behavior and duplicate its exception-propagation risks.

## Related Tests

- `sarek/interp/test/test_interp_error.ml` (or colocated equivalent): interpreter error formatting.
- `sarek/plugins/interpreter/test/test_interpreter_error.ml`: plugin error formatting.
- General Sarek value/type tests exercise supporting representations, but not full interpreter execution semantics.

## Missing Tests

- `LArrayElem` (plain-name lvalue) out-of-bounds write failure — still the one unchecked write path (`EArrayReadExpr`/`LArrayElemExpr` are covered by the 2026-07-02 fix above).
- Kernel declarations that include `DShared` or locals before/among `DParam` declarations.
- Too few and too many interpreter kernel arguments.
- Variant constructor hash collision behavior.
- Worker exception propagation in interpreter parallel execution.
- Equivalence between interpreter direct execution and plugin compatibility execution.

## Concrete Improvement/Fix Candidates

- Route the remaining unchecked `LArrayElem` write through the same checked-helper pattern already used by `EArrayRead`/`EArrayReadExpr`/`LArrayElemExpr` (raise `Interp_error.Array_bounds_error`).
- Replace `List.iter2` parameter binding with a declaration walker that consumes only `DParam` entries and validates leftover arguments.
- Assign stable constructor IDs during IR construction or carry constructor identity in the IR instead of hashing names modulo 256.
- Reuse the CPU runtime's fixed worker-exception mechanism once implemented, rather than maintaining a second pool implementation.
- Deprecate or harden legacy interpreter `Kernel.launch` so direct execution is the single authoritative path.
