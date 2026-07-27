# Sarek Types, Environments, Typing, And Monomorphization

## Component Inventory

This area covers `sarek/ppx/Sarek_types.ml`, `sarek/ppx/Sarek_scheme.ml`, `sarek/ppx/Sarek_typed_ast.ml`, `sarek/ppx/Sarek_env.ml`, `sarek/ppx/Sarek_typer.ml`, `sarek/ppx/Sarek_mono.ml`, and supporting primitive/registry files.

## Per-File Purpose

- `sarek/ppx/Sarek_types.ml`: type representation, fresh variables, unification, resolution, substitutions, stringification, and conversion from parsed type expressions.
- `sarek/ppx/Sarek_scheme.ml`: generalized type schemes for polymorphic functions and module items.
- `sarek/ppx/Sarek_typed_ast.ml`: typed expression, pattern, module item, and kernel records.
- `sarek/ppx/Sarek_env.ml`: variable/type/intrinsic/module environment, scopes, opens, and lookup order.
- `sarek/ppx/Sarek_typer.ml`: inference, unification checks, record/variant/type declaration handling, function application, loops, shared/superstep typing, and kernel typing.
- `sarek/ppx/Sarek_mono.ml`: specializes polymorphic module functions to concrete call-site types.
- `sarek/ppx/Sarek_core_primitives.ml`: built-in primitive signatures and convergence metadata.
- `sarek/ppx/Sarek_ppx_registry.ml`: compile-time registry feeding environment imports.

## Features And APIs

- Type variables and links support Hindley-Milner style inference in `sarek/ppx/Sarek_types.ml:49-176`.
- The standard environment imports core primitives and registered PPX intrinsics, then auto-opens GPU and Float32 modules in `sarek/ppx/Sarek_env.ml:278-349`.
- Type declarations register record fields and variant constructors in `sarek/ppx/Sarek_env.ml:119-145`.
- Record construction is typed in `sarek/ppx/Sarek_typer.ml:393-433`.
- Function application and intrinsic dispatch are typed in `sarek/ppx/Sarek_typer.ml:703-740`.
- Monomorphization collects polymorphic functions, call sites, and generated specialized copies in `sarek/ppx/Sarek_mono.ml:285-657`.

## Invariants

- Types should be fully resolved before lowering; the typed AST documents this at `sarek/ppx/Sarek_typed_ast.ml:27-31`.
- Built-in core primitive convergence metadata should survive module opens and shadowing; `open_module` preserves it in `sarek/ppx/Sarek_env.ml:183-259`.
- Record fields and constructors should resolve to the intended type declaration.
- Polymorphic functions should be specialized before lowering so residual type variables do not reach IR/native generation.

## Potential Invariant Violations Or Bugs

- **Needs KB/code decision — not resolved by this edit:** `TArr` unification ignores memory space in `sarek/ppx/Sarek_types.ml:127-131`. The code comment at `:129-130` ("Memspace may differ ... The actual memspace comes from create_array, not the type annotation") declares this intentional, while this KB entry has counted it as an invariant violation. These two positions disagree; do not silently resolve in either direction — a human must decide whether the KB invariant should be relaxed (documented erasure point) or the code comment is rationalizing a real gap that should be closed. Mirrored note in `kb/sarek/ppx/README.md`.
- Confirmed: `resolve_type` can leave unbound type variables unchanged in `sarek/ppx/Sarek_typed_ast.ml:159-164`, despite the resolved-type invariant.
- Confirmed: short record field and constructor names are stored in flat maps at `sarek/ppx/Sarek_env.ml:119-145`; later declarations can overwrite earlier names and make ambiguous fields/constructors order-dependent.
- Confirmed: record construction matches anonymous record types by exact field-name order in `sarek/ppx/Sarek_typer.ml:415-431`; same fields in a different order may fail or pick a different declaration.
- Confirmed: external or unknown record field access can defer with a fresh type variable and field index 0 in `sarek/ppx/Sarek_typer.ml:297-310`; field assignment has a similar deferred path at `sarek/ppx/Sarek_typer.ml:331-333`.
- Confirmed: unknown qualified functions are accepted as external fresh-type variables when the name contains `.` in `sarek/ppx/Sarek_typer.ml:689-695`; typos in qualified names may survive typing.
- **SPEC CHANGE 2026-07-02 (human decision, merged PRs #211/#213), commit `f8d436a9` — this is NOT a bug fix, do not read it as one.** Bare `float` in kernel type annotations previously mapped to `float64` (`sarek/ppx/Sarek_types.ml:322`, pre-change), conflicting with float32 assumptions elsewhere. Per explicit human decision recorded in the commit message ("keep float32 as the default numeric type for GPGPU kernels, not float64"), `type_of_type_expr` now maps `Sarek_ast.TEConstr ("float", [])` to `t_float32` (`sarek/ppx/Sarek_types.ml:319-325`, verified at HEAD `618768b7`), with an inline comment recording the rationale (GPU hardware executes float32 natively; float64 is slow/unsupported on much of it) and the escape hatch (use `float64` explicitly for double precision). This resolves the inconsistency the old text flagged: bare `float` in kernel type annotations now agrees with bare-`float`-literal typing (always float32) and with the legacy lowering/registration paths in `sarek/ppx/Sarek_lower.ml:116-135`/`sarek/ppx/Sarek_ppx.ml:101-107`, which already treated bare `float` as float32-sized before this change and were left untouched. **Citation update 2026-07-25 (task #78):** `Sarek_lower.ml` was retired when the dead legacy `Kirc_Ast` lowering was deleted, but the two functions this history points at (`c_type_of_core_type`, `typ_of_core_type`) were the file's surviving C-type half and now live — byte-identically, same line numbers — in `sarek/ppx/Sarek_ctype_gen.ml:116-139` (bare `float` → C `float` at `:120`, → `TReg Float32` at `:132`, re-verified 2026-07-25). The history is real and unchanged; only the module name moved. The companion `sarek/ppx/Sarek_ppx.ml:101-107` citation no longer resolves at current HEAD (that range is now an unrelated doc comment on `value_to_ocaml`) and was not re-anchored. Regression coverage: `sarek/tests/e2e/test_bare_float_is_float32.ml`. **Explicitly NOT changed by this decision:** `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:79-94` still maps `"float"` to `t_float64` — a separate intrinsic-*registration* type resolver that the `Sarek_float64`/`Float64.ml` stdlib relies on to mean float64 in its own signatures; changing it would corrupt the float64 math library. See `kb/sarek/ppx/intrinsics.md` and `kb/sarek/ppx/README.md`.

## Performance Or Maintainability Risks

- Flat field/constructor maps simplify lookup but make ambiguity handling fragile as registered and included type declarations grow.
- External qualified-name acceptance trades composability for delayed failures and makes typo detection weaker.
- Monomorphization mangle names encode normalized types in `sarek/ppx/Sarek_mono.ml:98-124`; unsupported residual `TVar` mangles as `"X"`, which can hide incomplete specialization.
- Type semantics are duplicated in parser conversion, intrinsic PPX conversion, IR lowering, native generation, and the C-type strings in `Sarek_ctype_gen`. (One duplicate fewer since 2026-07-25: the legacy lowering was deleted, task #78.)

## Related Tests

- `sarek/tests/unit/dune:1-30` includes `test_types`, `test_env`, `test_typer`, `test_scheme`, and `test_mono` (range re-verified 2026-07-25; shifted by one line when `test_lower` was dropped, task #78).
- `sarek/tests/e2e/dune:89-93` includes polymorphism, module polymorphism, inline recursion, and nested type E2E executables.
- Negative convention/type failures are listed in `sarek/tests/negative/dune:13-14` (line numbers shifted after the `test_tuple_param`/`test_fun_param` additions; see `kb/sarek/tests/negative.md`).

## Missing Tests

- Duplicate field/constructor names across two type declarations.
- Record construction with reordered fields.
- Qualified-name typo rejection behavior.
- Residual type variable detection after monomorphization.
- Memory-space mismatch typing for arrays.

## Concrete Improvement/Fix Candidates

- Use `(type_name, field_name)` and `(type_name, constructor_name)` keys internally, with explicit ambiguity errors for unqualified access.
- Add a post-typing assertion that no unresolved type variables reach lowerers unless explicitly allowed.
- Treat unknown qualified names as errors unless explicitly registered as external.
- Make field-order matching independent of source order.
