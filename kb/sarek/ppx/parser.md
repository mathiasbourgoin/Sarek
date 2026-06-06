# Sarek PPX Parser

## Component Inventory

Parser scope is primarily `sarek/ppx/Sarek_parse.ml` (the expression dispatcher and kernel/module-item parsing) and `sarek/ppx/Sarek_parse_helpers.ml` (extracted leaf helpers), with input/output shapes in `sarek/ppx/Sarek_ast.ml` and error reporting in `sarek/ppx/Sarek_error.ml`. `sarek/ppx/Sarek_ppx.ml` calls the parser from `expand_kernel` at `sarek/ppx/Sarek_ppx.ml:1430-1433`. The former monolithic `Sarek_parse.ml` (~839 lines) was reduced (pure move) by extracting leaf helpers into `Sarek_parse_helpers.ml` and decomposing `parse_expression`.

## Per-File Purpose

- `sarek/ppx/Sarek_ast.ml`: source AST for parsed kernels, including `type_expr`, `pattern`, `expr_desc`, `module_item`, and `kernel`.
- `sarek/ppx/Sarek_parse_helpers.ml`: extracted leaf helpers — `Parse_error_exn` (`sarek/ppx/Sarek_parse_helpers.ml:12`), `parse_type` (`:17`), pattern extractors (`extract_type_from_pattern`/`extract_name_from_pattern`/`extract_param_from_pattern`, `:66-83`), `parse_pattern` (`:104`), `parse_binop`/`parse_unop` (`:121`/`:145`), the AST-502 compatibility shims (`:162-188`), and `collect_fun_params` (`:211`).
- `sarek/ppx/Sarek_parse.ml`: maps `Parsetree` payloads into `Sarek_ast`. The `parse_expression` dispatcher (`sarek/ppx/Sarek_parse.ml:115`, now ~115 lines, was ~316) delegates five arm bodies to extracted helpers — `parse_assign_form` (`:332`), `parse_pragma_form` (`:343`), `parse_create_array_form` (`:365`), `parse_binop_or_app_form` (`:383`), and `parse_let_form` (`:399`); it also parses `let%shared`/`let%superstep` (`:36`/`:86`), kernel bodies, type declarations, and Sarek module items. Helper names are re-aliased from `Sarek_parse_helpers` at `sarek/ppx/Sarek_parse.ml:18-32`.
- `sarek/ppx/Sarek_error.ml`: carries parse failures as structured PPX errors.

## Features And APIs

- Parameter parsing requires `(name : type)` shapes in `sarek/ppx/Sarek_parse_helpers.ml:83-101` (`extract_param_from_pattern`).
- Core expression parser handles literals, variables, one-level qualified names, unary/binary ops, array/vector access, assignment, `if`, loops, `let`, local functions, record/variant construction, match, tuple, and extension nodes in `sarek/ppx/Sarek_parse.ml:115-327` (`parse_expression`), delegating five arm bodies to `parse_assign_form`/`parse_pragma_form`/`parse_create_array_form`/`parse_binop_or_app_form`/`parse_let_form` (`sarek/ppx/Sarek_parse.ml:332-453`).
- `let%shared` is parsed in `sarek/ppx/Sarek_parse.ml:36-81` (`parse_let_shared`).
- `let%superstep` and optional `[@divergent]` binding attributes are parsed in `sarek/ppx/Sarek_parse.ml:86-112` (`parse_superstep`).
- Kernel payloads and module items are parsed in `sarek/ppx/Sarek_parse.ml:454-631` (`parse_kernel_function`, `parse_module_items_from_structure`).

## Invariants

- Kernel parameters must be typed; unannotated parameters fail in `sarek/ppx/Sarek_parse_helpers.ml:89-95` (`extract_param_from_pattern`).
- Multi-argument variant constructors are rejected by the parser in `sarek/ppx/Sarek_parse_helpers.ml:44-65` (`parse_variant_constructors`, rejection at `:51`).
- Only syntax explicitly recognized by `parse_expression` should enter typing; unsupported forms end at `sarek/ppx/Sarek_parse.ml:327`.
- Source locations should be preserved well enough for user-facing errors.

## Potential Invariant Violations Or Bugs

- Probable: unsupported OCaml core types become `TEConstr ("unknown", [])` in `sarek/ppx/Sarek_parse_helpers.ml:17-31` (`parse_type`, fallback at `:31`) instead of failing at parse time; this can move diagnostics to later stages.
- Confirmed limitation: qualified identifiers support one module segment via `Ldot (Lident modname, name)` at `sarek/ppx/Sarek_parse.ml:139`; deeper module paths are not handled there.
- Confirmed limitation: standalone lambda expressions are rejected with a parse error directing the user to let-bound functions at `sarek/ppx/Sarek_parse.ml:283-288` (`parse_expression`, `is_function_expression` arm). (Prior KB text describing nested `ELet` construction was already stale before this refactor; the reject-and-redirect behavior is the moved-verbatim baseline.)
- Confirmed limitation: generic array syntax detection has an `is_array_access` helper that always returns false in `sarek/ppx/Sarek_parse.ml:449-452`; supported access paths depend on earlier explicit forms such as `Array.get`, `Array.set`, and `.%[]`.
- Confirmed limitation: mutable assignment only handles simple variable left-hand sides for `:=` in `sarek/ppx/Sarek_parse.ml:332-341` (`parse_assign_form`, error at `:340`).

## Performance Or Maintainability Risks

- Parser support is spread across many `Parsetree` shapes and has several syntax-specific paths for arrays, records, and module identifiers.
- Silent `unknown` type nodes make parser behavior harder to reason about because later failures may look like type errors rather than syntax support gaps.
- Location conversion uses partial position reconstruction in `sarek/ppx/Sarek_ast.ml:31-58`; this may reduce diagnostic precision.

## Related Tests

- `sarek/tests/unit/test_parse.ml:501-562` runs parser tests for operators, primitive/vector/arrow/tuple/var types, kernels, and basic expressions.
- E2E tests exercise PPX parsing through all executables in `sarek/tests/e2e/dune:56-93`.

## Missing Tests

- Deep qualified paths such as `A.B.f`.
- Direct lambda expressions inside kernels, either accepted with correct behavior or rejected clearly.
- Generic OCaml array syntax cases not routed through the explicit access forms.
- Unsupported OCaml type syntax diagnostics.
- `:=` assignment with non-variable lvalues, if intended to be rejected.

## Concrete Improvement/Fix Candidates

- Replace `TEConstr ("unknown", [])` fallback with a parse error carrying the unsupported OCaml type node.
- Add a recursive longident-to-name helper shared by variable, constructor, and intrinsic parsing.
- Either remove lambda parsing or represent lambdas explicitly and type them.
- Make `is_array_access` real, or delete it and document supported array syntax.
