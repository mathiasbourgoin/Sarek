# Sarek Intrinsics And Registration

## Component Inventory

Intrinsic support spans `sarek/ppx/Sarek_core_primitives.ml`, `sarek/ppx/Sarek_ppx_registry.ml`, `sarek/ppx/Sarek_env.ml`, `sarek/ppx/Sarek_native_intrinsics.ml`, `sarek/ppx/Sarek_quote.ml`, `sarek/ppx/Sarek_ppx.ml`, and `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`.

## Per-File Purpose

- `Sarek_core_primitives.ml`: built-in intrinsic names, types, and convergence metadata.
- `Sarek_ppx_registry.ml`: compile-time registry populated by stdlib/intrinsic modules and consumed by the PPX.
- `Sarek_env.ml`: imports registered intrinsics into the typing environment.
- `Sarek_native_intrinsics.ml`: generates native fallback OCaml expressions for intrinsic constants/functions.
- `Sarek_quote.ml`: keeps runtime intrinsic references alive in generated kernels.
- `Sarek_ppx.ml`: initializes stdlib/Float64 registration and handles Sarek type/module/include expansion.
- `Sarek_ppx_intrinsic.ml`: authoring PPX for `%sarek_intrinsic` and `%sarek_extend`.

## Features And APIs

- The PPX rewriter forces stdlib and Float64 initialization at `sarek/ppx/Sarek_ppx.ml:21-28`.
- Environment import of registry intrinsics happens in `sarek/ppx/Sarek_env.ml:278-349`.
- `%sarek_intrinsic` type and function registration require `device`, `ctype`, and/or `ocaml` fields in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:173-387`.
- `%sarek_extend` tries to register an external OCaml function for Sarek use in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:442-478`.
- Intrinsic PPX module-path inference is in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:61-76`.

## Invariants

- Runtime and compile-time registries must receive equivalent type/function metadata.
- Intrinsic type parsing must agree with the main Sarek parser and typer.
- Device names and OCaml implementations must be registered under the same Sarek name that kernels resolve.
- Native fallback must implement every intrinsic admitted by the typer and convergence strategy.

## Potential Invariant Violations Or Bugs

- Confirmed: intrinsic PPX type parser supports only simple, array, vector, and arrow forms and falls back to `unknown` in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:24-43`.
- Confirmed: `"float"` maps to `float64` in intrinsic PPX conversion at `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:79-94`, which conflicts with some lower/register paths.
- Confirmed: array arguments in the intrinsic PPX are always `TArr Local` in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:106-121`.
- Probable: module-path inference from the source file directory in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:61-76` is heuristic and can be wrong for nested or wrapped library layouts.
- Probable: `%sarek_extend` expects `Ppat_var` and then splits the variable name on dots in `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml:453-462`; ordinary OCaml variable patterns cannot contain dots, so documented dotted names may not parse as intended.

## Performance Or Maintainability Risks

- Intrinsic registration writes both runtime and PPX registry code; drift can create compile-time success with runtime failure or the reverse.
- The intrinsic PPX duplicates type parsing logic rather than reusing `Sarek_parse`/`Sarek_types`.
- Module path heuristics make behavior sensitive to Dune layout and generated-file paths.

## Related Tests

- Unit intrinsic tests include `test_core_primitives`, `test_native_intrinsics`, and `test_ppx_registry` in `sarek/tests/unit/dune:8-21`.
- E2E intrinsic coverage includes math and bitwise executables in `sarek/tests/e2e/dune:81-82`.
- Negative warp/divergence behavior involving intrinsics is documented in `sarek/tests/negative/dune:11`.

## Missing Tests

- `%sarek_extend Module.func` syntax and generated registration.
- Intrinsic PPX unsupported type syntax diagnostics.
- Intrinsic array memory-space semantics.
- Runtime/PPX registry parity for each intrinsic definition.
- Wrapped/nested library module-path inference.

## Concrete Improvement/Fix Candidates

- Share type parsing/conversion between the main PPX and intrinsic PPX.
- Require explicit module/name metadata for intrinsic registration instead of deriving it from file paths.
- Replace `unknown` fallback with structured extension errors.
- Add compile tests for `%sarek_intrinsic` and `%sarek_extend` generated code.
