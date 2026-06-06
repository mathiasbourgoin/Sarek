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

- Confirmed: `TArr` unification ignores memory space in `sarek/ppx/Sarek_types.ml:128-131`.
- Confirmed: `resolve_type` can leave unbound type variables unchanged in `sarek/ppx/Sarek_typed_ast.ml:159-164`, despite the resolved-type invariant.
- Confirmed: short record field and constructor names are stored in flat maps at `sarek/ppx/Sarek_env.ml:119-145`; later declarations can overwrite earlier names and make ambiguous fields/constructors order-dependent.
- Confirmed: record construction matches anonymous record types by exact field-name order in `sarek/ppx/Sarek_typer.ml:415-431`; same fields in a different order may fail or pick a different declaration.
- Confirmed: external or unknown record field access can defer with a fresh type variable and field index 0 in `sarek/ppx/Sarek_typer.ml:297-310`; field assignment has a similar deferred path at `sarek/ppx/Sarek_typer.ml:331-333`.
- Confirmed: unknown qualified functions are accepted as external fresh-type variables when the name contains `.` in `sarek/ppx/Sarek_typer.ml:689-695`; typos in qualified names may survive typing.
- Confirmed: bare `float` maps to `float64` in `sarek/ppx/Sarek_types.ml:322`, conflicting with some later float32 assumptions.

## Performance Or Maintainability Risks

- Flat field/constructor maps simplify lookup but make ambiguity handling fragile as registered and included type declarations grow.
- External qualified-name acceptance trades composability for delayed failures and makes typo detection weaker.
- Monomorphization mangle names encode normalized types in `sarek/ppx/Sarek_mono.ml:98-124`; unsupported residual `TVar` mangles as `"X"`, which can hide incomplete specialization.
- Type semantics are duplicated in parser conversion, intrinsic PPX conversion, legacy lowering, IR lowering, and native generation.

## Related Tests

- `sarek/tests/unit/dune:1-32` includes `test_types`, `test_env`, `test_typer`, `test_scheme`, and `test_mono`.
- `sarek/tests/e2e/dune:89-93` includes polymorphism, module polymorphism, inline recursion, and nested type E2E executables.
- Negative convention/type failures are listed in `sarek/tests/negative/dune:14-15`.

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
