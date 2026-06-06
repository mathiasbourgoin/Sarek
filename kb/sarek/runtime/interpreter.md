# Interpreter Runtime

## Component Inventory

Reviewed interpreter runtime files: the `Sarek_ir_interp*` module family, `sarek/sarek/Interp_error.ml`, interpreter-facing pieces of `sarek/sarek/Sarek_ir.ml`, and `sarek/plugins/interpreter/**`. Plugin-specific notes are also summarized in [plugins.md](plugins.md). The former monolithic `Sarek_ir_interp.ml` (~1573 lines) was split (pure move) into four modules.

## Per-File Purpose

- `sarek/sarek/Sarek_ir_interp_value.ml`: the `value` type, environment, thread state, `to_*` conversions, `eval_binop`/`eval_unop`, and path predicates (`sarek/sarek/Sarek_ir_interp_value.ml:10-280`).
- `sarek/sarek/Sarek_ir_interp_intrinsics.ml`: GPU/float/int/type-conversion intrinsics, split out of the former monolithic `eval_intrinsic` (`sarek/sarek/Sarek_ir_interp_intrinsics.ml:14-323`).
- `sarek/sarek/Sarek_ir_interp_eval.ml`: the recursive `eval_expr`/exec chain over expressions, statements, and lvalues (`sarek/sarek/Sarek_ir_interp_eval.ml`).
- `sarek/sarek/Sarek_ir_interp.ml` (reduced): interpreter-local `DomainPool`, the `run_grid*` drivers, and the public `run_kernel*` API (`sarek/sarek/Sarek_ir_interp.ml:78-535`).
- `sarek/sarek/Interp_error.ml`: interpreter error variants and formatting.
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

- Bounds checks are inconsistent. `EArrayRead` checks bounds at `sarek/sarek/Sarek_ir_interp_eval.ml:55-62`, but `EArrayReadExpr` directly indexes with `a.(i)` at `sarek/sarek/Sarek_ir_interp_eval.ml:63-72`. Assignment through `LArrayElem` and `LArrayElemExpr` also writes directly at `sarek/sarek/Sarek_ir_interp_eval.ml:350-363`.
- `run_kernel` binds declarations and arguments with `List.iter2` at `sarek/sarek/Sarek_ir_interp.ml:202-208`. If declarations include non-parameter entries or the argument count differs, callers can get a generic `Invalid_argument` or wrong binding. The later `args_from_kernel_args` path at `sarek/sarek/Sarek_ir_interp.ml:469-528` is more explicit.
- Variant tags are computed from `Hashtbl.hash ctor mod 256` when constructing and matching variants at `sarek/sarek/Sarek_ir_interp_eval.ml:126-127`, `sarek/sarek/Sarek_ir_interp_eval.ml:151`, and `sarek/sarek/Sarek_ir_interp_eval.ml:286`. Constructor collisions are possible.
- Interpreter-local `DomainPool.worker` catches and discards task exceptions at `sarek/sarek/Sarek_ir_interp.ml:92-110`, especially `sarek/sarek/Sarek_ir_interp.ml:105` (`try task () with _ -> ()`), then signals completion.

## Performance Or Maintainability Risks

- Direct interpreter execution and plugin legacy `Kernel.launch` have different argument paths, increasing semantic drift.
- The evaluator has many expression cases with manual recursion; new expression variants can miss checks or substitution-like logic.
- Hash-derived variant tags make debugging and cross-run compatibility harder than explicit constructor IDs.
- Interpreter-local parallel helpers duplicate CPU runtime pool behavior and duplicate its exception-propagation risks.

## Related Tests

- `sarek/sarek/test/test_interp_error.ml`: interpreter error formatting.
- `sarek/plugins/interpreter/test/test_interpreter_error.ml`: plugin error formatting.
- General Sarek value/type tests under `sarek/sarek/test/**` exercise supporting representations, but not full interpreter execution semantics.

## Missing Tests

- `EArrayReadExpr`, `LArrayElem`, and `LArrayElemExpr` out-of-bounds failures.
- Kernel declarations that include `DShared` or locals before/among `DParam` declarations.
- Too few and too many interpreter kernel arguments.
- Variant constructor hash collision behavior.
- Worker exception propagation in interpreter parallel execution.
- Equivalence between interpreter direct execution and plugin compatibility execution.

## Concrete Improvement/Fix Candidates

- Route every array read/write through shared checked helpers that raise `Interp_error.Array_bounds_error`.
- Replace `List.iter2` parameter binding with a declaration walker that consumes only `DParam` entries and validates leftover arguments.
- Assign stable constructor IDs during IR construction or carry constructor identity in the IR instead of hashing names modulo 256.
- Reuse the CPU runtime's fixed worker-exception mechanism once implemented, rather than maintaining a second pool implementation.
- Deprecate or harden legacy interpreter `Kernel.launch` so direct execution is the single authoritative path.
