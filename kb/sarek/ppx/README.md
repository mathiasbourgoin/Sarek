# Sarek PPX Compiler Knowledge Base

## Component Inventory

The PPX compiler lives in `sarek/ppx/` with the intrinsic authoring PPX in `sarek/ppx_intrinsic/`. The compiler is split across two libraries plus a facade (line ranges re-verified 2026-07-25 after task #78): the frontend library `sarek_frontend` at `sarek/ppx/dune:1-35`, the quote/native-gen library `sarek_native_gen` at `sarek/ppx/dune:37-54`, and the `sarek_ppx_lib` facade that `re_export`s both at `sarek/ppx/dune:61-67`. The installed rewriter `sarek_ppx` is declared at `sarek/ppx/dune:69-78`.

The compilation path is:

1. `sarek/ppx/Sarek_parse.ml` (with leaf helpers in `sarek/ppx/Sarek_parse_helpers.ml`) parses OCaml AST payloads into `Sarek_ast`.
2. `sarek/ppx/Sarek_typer.ml` resolves names and infers `Sarek_typed_ast`.
3. `sarek/ppx/Sarek_mono.ml` specializes polymorphic module functions.
4. `sarek/ppx/Sarek_convergence.ml` rejects unsafe barriers/warp collectives and selects execution mode.
5. `sarek/ppx/Sarek_tailrec*.ml` rewrites tail-recursive functions or inlines approved non-tail recursion.
6. `sarek/ppx/Sarek_lower_ir.ml` lowers typed kernels to `Sarek_ir_ppx`.
7. `sarek/ppx/Sarek_quote_ir.ml` and `sarek/ppx/Sarek_quote.ml` quote runtime values and generated wrappers.
8. The `sarek/ppx/Sarek_native_gen*` modules (`Sarek_native_gen_base`, `Sarek_native_gen_expr`, `Sarek_native_gen`, `Sarek_native_gen_kernel`) emit the CPU/native fallback function.

`sarek/ppx/Sarek_ppx.ml:1732` (`let expand_kernel ~ctxt payload`) is the top-level `expand_kernel` pipeline entry point (re-verified 2026-07-25; prior `:1416` anchor had drifted, as had `:1387-1542` before it — this is unrelated to task #78, which changed only one line of this file). It initializes the stdlib and Float64 registrations at `sarek/ppx/Sarek_ppx.ml:22-27`, scans the current file for `[@@sarek.type]`/`[@sarek.module]` registrations before parse at `sarek/ppx/Sarek_ppx.ml:1738-1760` onward, then runs parse/type/convergence/mono/tailrec/lower/quote. This is the *only* lowering/quoting path since 2026-07-25 (task #78): `expand_kernel` → `Sarek_lower_ir.lower_kernel` → `Sarek_quote_ir`.

## Per-File Purpose

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
- `sarek/ppx/Sarek_ctype_gen.ml`: C struct/union/builder source-string generation for `[@@sarek.type]` top-level registrations (`mangle_type_name`, `c_type_of_typ`, `record_constructor_strings`, `variant_constructor_strings`, `c_type_of_core_type`, `typ_of_core_type`, `constructor_strings_of_core_type_decl`). Its only caller is `expand_sarek_type` in `sarek/ppx/Sarek_ppx.ml:2184` onward (`Sarek_ctype_gen.constructor_strings_of_core_type_decl` at `:2189`). Created 2026-07-25 (task #78) by renaming `Sarek_lower.ml` down to this surviving half; the legacy `Kirc_Ast` lowering that made up the rest of that file was deleted.
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
- `[@@sarek.type]` and `[@@sarek.type private]` type registration, accessor generation, runtime type registration, and PPX registry registration: `expand_sarek_type` at `sarek/ppx/Sarek_ppx.ml:2184-2200` (re-verified 2026-07-25; prior `:1336-1380` range was stale). It obtains the C struct/union/builder strings from `Sarek_ctype_gen.constructor_strings_of_core_type_decl` (`:2189`) and splices a `Sarek.Kirc_types.register_constructor_string` loop after the type declaration.
- `[@sarek.module]` module-local constants/functions/types and module-open handling in `sarek/ppx/Sarek_ppx.ml:2204-2270` (re-verified 2026-07-25; prior `:1591-1668` range was stale).
- `[%sarek_include "..."]` include scanning: `expand_sarek_include` at `sarek/ppx/Sarek_ppx.ml:2302` (re-verified 2026-07-25; prior `:1711` anchor, and `:1670-1818` before it, were stale).
- `%sarek_intrinsic` and `%sarek_extend` registration extensions in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`.

## Invariants

- Kernel parameters must have explicit annotations; parser enforcement is in `sarek/ppx/Sarek_parse_helpers.ml:83-95` (`extract_param_from_pattern`).
- Typed AST node `ty` values are intended to be resolved, documented in `sarek/ppx/Sarek_typed_ast.ml:27-31`.
- Environment lookups prefer local variables, then intrinsic constants/functions, constructors, and local functions in `sarek/ppx/Sarek_env.ml:360-375`.
- Barriers and warp collectives must not appear in diverged control flow, enforced by `sarek/ppx/Sarek_convergence.ml:135-244`.
- Non-tail recursion is only accepted through explicit inline pragmas; tail recursion is converted to loops in `sarek/ppx/Sarek_tailrec.ml:51-166`.

## Potential Invariant Violations Or Bugs

- **SPEC CHANGE 2026-07-02 (human decision, merged PRs #211/#213), commit `f8d436a9` — NOT a bug fix.** Bare `float` used to be inconsistent across stages: `Sarek_types.ml:322` mapped it to `float64` while legacy lowering/registering already mapped it as float32-sized in `sarek/ppx/Sarek_lower.ml:116-135`/`sarek/ppx/Sarek_ppx.ml:101-107`. **Citation update 2026-07-25 (task #78):** `Sarek_lower.ml` was retired; the two functions carrying that float32-sized registration mapping (`c_type_of_core_type`, `typ_of_core_type`) survive byte-identically — including their line numbers — in the renamed `sarek/ppx/Sarek_ctype_gen.ml:116-139` (bare `float` → C `float` at `:120`, → `TReg Float32` at `:132`, re-verified 2026-07-25). The companion `sarek/ppx/Sarek_ppx.ml:101-107` citation no longer resolves at current HEAD (that range is now an unrelated doc comment on `value_to_ocaml`) and was not re-anchored. Per explicit human decision ("keep float32 as the default numeric type for GPGPU kernels, not float64"), `sarek/ppx/Sarek_types.ml:319-325` (verified HEAD `618768b7`) now maps bare `float` in kernel type annotations to `t_float32`, aligning it with the legacy paths and with float-literal typing (always float32). Detail and regression test in `kb/sarek/ppx/types.md`. **Deliberately unaffected:** `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:79-94` still treats `"float"` as `float64` — a separate intrinsic-registration resolver the `Sarek_float64` stdlib depends on; see `kb/sarek/ppx/intrinsics.md`.
- **Needs KB/code decision — not resolved by this edit:** array memory space is ignored during `TArr` unification at `sarek/ppx/Sarek_types.ml:127-131`, so local/shared/global array types can unify despite different memory spaces. The code comment at `:129-130` states this is intentional ("Memspace may differ ... The actual memspace comes from create_array, not the type annotation"), while the KB has historically counted this as an invariant violation. These two positions disagree and a human must decide which is authoritative: either (a) the KB invariant is wrong and should be relaxed/annotated as accepted erasure, or (b) the code comment is rationalizing a real gap and unification should be tightened. Do not silently pick a side — see `kb/sarek/ppx/types.md` for the mirrored note.
- **FIXED 2026-07-02 (merged, PRs #211/#213), commit `06a670ea`, partially — see fd-leak caveat below.** Previously: `scan_file_for_sarek_types` wrapped its whole body in `with _ -> ()`, silently discarding the scanned file's name and the exception on any failure (unreadable file, parse error, or a malformed `[@sarek.*]` declaration). Verified fixed at HEAD `618768b7`: `sarek/ppx/Sarek_ppx.ml:256-352` now catches the exception, prints a diagnostic naming the scanned file and the exception via `Printf.eprintf` (`:349-352`), and returns `Some diagnostic` instead of silently returning `()` — the triggering file's own compilation still succeeds (an `[@ocaml.ppwarning]` approach was tried and rejected because this project's dune "dev" profile promotes warning 22 to a hard error, per the in-code doc comment at `:220-252`). Scanning of other files is unaffected since each scan call is independently guarded. Regression coverage: `sarek/tests/unit/test_ppx_scan_diagnostics.ml`, `sarek/tests/e2e/test_ppx_scan_failure_warning.ml`.
  - **fd-leak: FIXED 2026-07-02 (merged, PR #215), commit `b37411b2`.** The fd-leak fix was originally made on #211 but lost in a rebase/force-push (a stale local branch was rebased over the pushed fix); it was re-applied and merged separately. Verified: `scan_file_for_sarek_types` now wraps the lexbuf setup + `Parse.implementation` in `Fun.protect ~finally:(fun () -> close_in_noerr ic)`, so the descriptor is closed whether or not the parse raises — the malformed-file path (the case the graceful handler exists for) no longer leaks.
- **FIXED (2026-07-02 audit):** convergence is now dataflow-sensitive — `varying_vars` tracking (`sarek/ppx/Sarek_convergence.ml:43-45, 89, 108-111`) propagates thread-varying status through `let`/`let mut` bindings into later branch checks (commits `5e0cd683`, `c5c39c71`). See `kb/sarek/ppx/convergence.md` for detail. Stale artifact: the file's own header comment at `sarek/ppx/Sarek_convergence.ml:27` still lists dataflow analysis under "Future extensions" and was not updated when the feature landed.
- Probable: simple native execution accepts only selected global/thread ids, but dimensionality analysis can classify `global_size_x` as simple at `sarek/ppx/Sarek_convergence.ml:452-460`; native intrinsic generation rejects it in simple modes at `sarek/ppx/Sarek_native_intrinsics.ml:191-218`.

## Performance Or Maintainability Risks

- Multiple parallel type-name models exist: `Sarek_types`, `Sarek_ir_ppx`, runtime IR, native OCaml AST generation, the C-type strings in `Sarek_ctype_gen`, and intrinsic PPX parsing. This raises drift risk; the `float` mismatch is an example. (One model fewer since 2026-07-25: the legacy `Kirc_Ast` mirror was deleted by task #78.)
- `Sarek_ppx.ml` is large and mixes scanning, registration, payload expansion, error handling, and rewriter registration.
- **RESOLVED 2026-07-25 (task #78).** Previously: `sarek/ppx/Sarek_lower.ml` was legacy but still tested and exposed through helpers, so semantic drift between legacy and IR/native lowering could survive. The legacy lowering and `Kirc_Ast` were proven unreachable (`Sarek_quote.quote_elttype` emitted `Sarek.Kirc_Ast.EInt32`, a module the `sarek` runtime library never had, so the generated code could not have compiled; 0 of 85 PPX-expanded artifacts referenced `Kirc_Ast`) and deleted. Only one lowering path remains — `Sarek_ppx.expand_kernel` → `Sarek_lower_ir.lower_kernel` → `Sarek_quote_ir` — so there is no longer a second semantics to drift from. See `kb/sarek/ppx/lowering.md`.
- Monomorphization and tailrec inlining can grow ASTs; tests cover some limits, but final generated size is not globally bounded.

## Related Tests

- Unit test inventory is in `sarek/tests/unit/dune:1-30` (re-verified 2026-07-25; shifted by one line when `test_lower` was dropped, task #78).
- E2E PPX tests are listed in `sarek/tests/e2e/dune:56-145` and run via `sarek/tests/e2e/dune:147-181`.
- Negative compile tests document expected errors in `sarek/tests/negative/dune:8-18` (updated 2026-07-02: `test_tuple_param`/`test_fun_param` were appended, extending the range from the previous `:8-16`). See `kb/sarek/tests/negative.md`.
- PPX-local tests live under `sarek/ppx/test/`, including `test_sarek_error.ml`, `test_sarek_reserved.ml`, and `test_sarek_debug.ml`.

## Missing Tests

- Shared/local/global array memory-space mismatches.
- Aliased or module-qualified barrier/warp intrinsic calls feeding indirect divergence (the direct `let tid = ... in if tid > 0 then barrier()` case is now covered by the `varying_vars` dataflow pass — see convergence.md).
- Kernel-local helper functions that construct/match a module-level `[@@sarek.type]` variant: `gen_module_fun` and `core_type_of_typ` build their conversion context without a `current_module`, so `is_same_module` always fails inside a helper and the variant is over-qualified (`Test_x.shape`/`.Circle`), which becomes a self-reference `ocamlopt` rejects once dune wraps the unit as `Dune__exe__Test_x`. Confirmed 2026-07-02 (why `test_klet_variant` inlines its match instead of using a helper). Real, reproducible; fix needs `current_module` threaded into both context builders in `sarek/ppx/Sarek_native_gen.ml`/`Sarek_native_intrinsics.ml`.
- Native simple-mode kernels using `global_size_x`, `global_size_y`, or `global_size_z`.
- (Resolved, no longer missing: bare-`float` consistency across kernel-body stages — now spec-defined as float32 by human decision, see types.md; include-scan failures surfacing instead of silently swallowing, and the fd-leak on the parse-error path — both fixed, see above.)

## Concrete Improvement/Fix Candidates

- Resolve the `TArr` memspace-unification disagreement between the code comment (intentional) and the KB invariant (violation) — human decision needed, see the note above and `types.md`.
- Thread `current_module` into `gen_module_fun`/`core_type_of_typ` so kernel-local helpers can construct/match module-level `[@@sarek.type]` variants without over-qualification (see the helper-over-qualification entry under Potential Invariant Violations).
- Update the stale "Future extensions: Dataflow analysis" line in `Sarek_convergence.ml:27` now that `varying_vars` dataflow is implemented.
- Decide whether simple native mode supports `global_size_*`; either implement it or force full execution mode when those constants appear.
