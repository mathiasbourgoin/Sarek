# Sarek PPX Compiler Knowledge Base

## Component Inventory

The PPX compiler lives in `sarek/ppx/` with the intrinsic authoring PPX in `sarek/ppx_intrinsic/`. The main library is declared in `sarek/ppx/dune:1-38`; the installed rewriter is declared in `sarek/ppx/dune:40-49`.

The compilation path is:

1. `sarek/ppx/Sarek_parse.ml` (with leaf helpers in `sarek/ppx/Sarek_parse_helpers.ml`) parses OCaml AST payloads into `Sarek_ast`.
2. `sarek/ppx/Sarek_typer.ml` resolves names and infers `Sarek_typed_ast`.
3. `sarek/ppx/Sarek_mono.ml` specializes polymorphic module functions.
4. `sarek/ppx/Sarek_convergence.ml` rejects unsafe barriers/warp collectives and selects execution mode.
5. `sarek/ppx/Sarek_tailrec*.ml` rewrites tail-recursive functions or inlines approved non-tail recursion.
6. `sarek/ppx/Sarek_lower_ir.ml` lowers typed kernels to `Sarek_ir_ppx`.
7. `sarek/ppx/Sarek_quote_ir.ml` and `sarek/ppx/Sarek_quote.ml` quote runtime values and generated wrappers.
8. The `sarek/ppx/Sarek_native_gen*` modules (`Sarek_native_gen_base`, `Sarek_native_gen_expr`, `Sarek_native_gen`, `Sarek_native_gen_kernel`) emit the CPU/native fallback function.

`sarek/ppx/Sarek_ppx.ml:1387-1542` is the top-level `expand_kernel` pipeline. It initializes the stdlib and Float64 registrations at `sarek/ppx/Sarek_ppx.ml:21-28`, scans module/include registrations before parse at `sarek/ppx/Sarek_ppx.ml:1393-1443`, then runs parse/type/convergence/mono/tailrec/lower/quote.

## Per-File Purpose

- `sarek/ppx/Kirc_Ast.ml`: legacy Kirc AST mirror still used by legacy lowering and compatibility tests.
- `sarek/ppx/Sarek_ast.ml`: source-level kernel AST, including source locations, memory spaces, patterns, expressions, type declarations, module items, and kernel records.
- `sarek/ppx/Sarek_typed_ast.ml`: typed AST with resolved types, type schemes, and typed kernel/module item shapes.
- `sarek/ppx/Sarek_types.ml`: core type representation, type variables, unification, substitution, and OCaml type conversion.
- `sarek/ppx/Sarek_scheme.ml`: polymorphic scheme helpers used by typing and monomorphization.
- `sarek/ppx/Sarek_core_primitives.ml`: built-in constants/functions, convergence metadata, and primitive type signatures.
- `sarek/ppx/Sarek_ppx_registry.ml`: compile-time registry for Sarek modules, types, intrinsics, and includes.
- `sarek/ppx/Sarek_env.ml`: lexical/type/module environment and lookup policy.
- `sarek/ppx/Sarek_error.ml`: structured errors and result helpers.
- `sarek/ppx/Sarek_reserved.ml`: C/CUDA/OpenCL reserved identifier checks.
- `sarek/ppx/Sarek_parse.ml`: PPX payload parser for kernels and module items (the `parse_expression` dispatcher plus kernel/module-item parsing).
- `sarek/ppx/Sarek_parse_helpers.ml`: extracted leaf parser helpers (`parse_type`, pattern extractors, `parse_pattern`, binop/unop parsers, AST-502 compat shims, `collect_fun_params`, `Parse_error_exn`).
- `sarek/ppx/Sarek_typer.ml`: type inference, type declaration registration, name resolution, and typed kernel production.
- `sarek/ppx/Sarek_mono.ml`: monomorphization for polymorphic module functions.
- `sarek/ppx/Sarek_convergence.ml`: barrier/warp collective safety checks and execution strategy inference.
- `sarek/ppx/Sarek_tailrec*.ml`: tail recursion analysis, loop conversion, bounded/inlined recursion support, and pragma parsing.
- `sarek/ppx/Sarek_lower.ml`: legacy lowering to `Kirc_Ast`.
- `sarek/ppx/Sarek_ir_ppx.ml`: compile-time mirror of runtime Sarek IR.
- `sarek/ppx/Sarek_lower_ir.ml`: typed AST to Sarek IR lowering.
- `sarek/ppx/Sarek_quote_ir.ml`: quoted OCaml expression generation for Sarek IR.
- `sarek/ppx/Sarek_quote.ml`: full quoted kernel/runtime wrapper generation.
- `sarek/ppx/Sarek_native_helpers.ml`: location, name, default value, and helper utilities for native generation.
- `sarek/ppx/Sarek_native_intrinsics.ml`: native OCaml expressions for Sarek intrinsic constants/functions.
- `sarek/ppx/Sarek_native_gen_base.ml`: native-gen context/types, name helpers, `gen_literal`/`gen_variable`.
- `sarek/ppx/Sarek_native_gen_expr.ml`: the `~gen_expr`-parameterised sub-generators (memory access, let bindings, control flow, data structures, special exprs, BSP).
- `sarek/ppx/Sarek_native_gen.ml`: reduced core — recursive `gen_expr_impl`, public entry points, and module/type-declaration generation.
- `sarek/ppx/Sarek_native_gen_kernel.ml`: argument casting, the types object, and CPU kernel builders.
- `sarek/ppx/Sarek_debug.ml`: opt-in debug logging.
- `sarek/ppx/Sarek_ppx.ml`: registered PPX rewriter and top-level Sarek syntax/type/module transformations.
- `sarek/ppx/test/*`: local PPX unit tests for reserved words, errors, and debug helpers.
- `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`: PPX for defining/registering Sarek intrinsics and extending external functions.

## Features And APIs

- `[%kernel fun ... -> ...]` and `let%kernel` expansion to runtime Sarek kernels.
- `[@@sarek.type]` and `[@@sarek.type private]` type registration, accessor generation, runtime type registration, and PPX registry registration in `sarek/ppx/Sarek_ppx.ml:1336-1380`.
- `[@sarek.module]` module-local constants/functions/types and module-open handling in `sarek/ppx/Sarek_ppx.ml:1591-1668`.
- `[%sarek_include "..."]` include scanning in `sarek/ppx/Sarek_ppx.ml:1670-1818`.
- `%sarek_intrinsic` and `%sarek_extend` registration extensions in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`.

## Invariants

- Kernel parameters must have explicit annotations; parser enforcement is in `sarek/ppx/Sarek_parse_helpers.ml:83-95` (`extract_param_from_pattern`).
- Typed AST node `ty` values are intended to be resolved, documented in `sarek/ppx/Sarek_typed_ast.ml:27-31`.
- Environment lookups prefer local variables, then intrinsic constants/functions, constructors, and local functions in `sarek/ppx/Sarek_env.ml:360-375`.
- Barriers and warp collectives must not appear in diverged control flow, enforced by `sarek/ppx/Sarek_convergence.ml:135-244`.
- Non-tail recursion is only accepted through explicit inline pragmas; tail recursion is converted to loops in `sarek/ppx/Sarek_tailrec.ml:51-166`.

## Potential Invariant Violations Or Bugs

- Confirmed: bare `float` is inconsistent across stages. `sarek/ppx/Sarek_types.ml:322` maps it to `float64`, while legacy lowering/registering maps it as float32-sized in `sarek/ppx/Sarek_lower.ml:116-135` and `sarek/ppx/Sarek_ppx.ml:101-107`; `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:79-94` again treats it as `float64`.
- Confirmed: array memory space is ignored during unification at `sarek/ppx/Sarek_types.ml:128-131`, so local/shared/global array types can unify despite different memory spaces.
- Confirmed: include/module source scanning suppresses all exceptions in `sarek/ppx/Sarek_ppx.ml:208-305`; malformed or unreadable files can silently drop registrations.
- Confirmed limitation: convergence is not dataflow-sensitive. The file header calls this out in `sarek/ppx/Sarek_convergence.ml:22-29`, and `is_thread_varying` only recognizes direct terminal expressions in `sarek/ppx/Sarek_convergence.ml:79-121`.
- Probable: simple native execution accepts only selected global/thread ids, but dimensionality analysis can classify `global_size_x` as simple at `sarek/ppx/Sarek_convergence.ml:452-460`; native intrinsic generation rejects it in simple modes at `sarek/ppx/Sarek_native_intrinsics.ml:191-218`.

## Performance Or Maintainability Risks

- Multiple parallel type-name models exist: `Sarek_types`, legacy `Kirc_Ast`, `Sarek_ir_ppx`, runtime IR, native OCaml AST generation, and intrinsic PPX parsing. This raises drift risk; the `float` mismatch is an example.
- `Sarek_ppx.ml` is large and mixes scanning, registration, payload expansion, error handling, and rewriter registration.
- `sarek/ppx/Sarek_lower.ml` is legacy but still tested and exposed through helpers, so semantic drift between legacy and IR/native lowering can survive.
- Monomorphization and tailrec inlining can grow ASTs; tests cover some limits, but final generated size is not globally bounded.

## Related Tests

- Unit test inventory is in `sarek/tests/unit/dune:1-32`.
- E2E PPX tests are listed in `sarek/tests/e2e/dune:56-145` and run via `sarek/tests/e2e/dune:147-181`.
- Negative compile tests document expected errors in `sarek/tests/negative/dune:8-16`.
- PPX-local tests live under `sarek/ppx/test/`, including `test_sarek_error.ml`, `test_sarek_reserved.ml`, and `test_sarek_debug.ml`.

## Missing Tests

- Bare `float` consistency across parser, typer, lowering, intrinsic PPX, native generation, and runtime registration.
- Shared/local/global array memory-space mismatches.
- Include scanning failures that should be surfaced instead of swallowed.
- Indirect convergence violations such as `let tid = thread_idx_x in if tid > 0 then block_barrier ()`.
- Native simple-mode kernels using `global_size_x`, `global_size_y`, or `global_size_z`.

## Concrete Improvement/Fix Candidates

- Define one authoritative mapping for surface `float`; update parser, type conversion, registration size, native generation, and intrinsic PPX to use it.
- Include `memspace` in `TArr` unification or document and enforce the intended erasure point.
- Replace broad `with _ -> ()` include scanning with structured diagnostics and tests.
- Add a minimal thread-varying dataflow pass before barrier checks.
- Decide whether simple native mode supports `global_size_*`; either implement it or force full execution mode when those constants appear.
