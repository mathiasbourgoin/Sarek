# Intake + Plan — parse-split (Intake 6, two steps)

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

`sarek/ppx/Sarek_parse.ml` (839 lines, no `.mli`). Two steps:
- **6a (this brief):** pure-move split — extract leaf helpers into
  `Sarek_parse_helpers.ml`. Mechanical, byte-identical, test-gated.
- **6b (separate):** decompose the 316-line `parse_expression` (47 match arms) into
  category sub-parsers via open recursion. Logic refactor, test-gated + reviewed.

## External API (must stay reachable as `Sarek_parse.<name>`)

Used outside the module: `Parse_error_exn`, `extract_name_from_pattern`,
`extract_param_from_pattern`, `extract_type_from_pattern`, `collect_fun_params`,
`pattern_of_param`, `parse_expression`, `parse_payload`.

## Step 6a — module DAG: helpers ← Sarek_parse (main)

**`Sarek_parse_helpers.ml`** (leaf) — move lines ~15–258:
`loc_of_ppxlib`, `exception Parse_error_exn`, `loc_to_sloc`, `parse_type`,
`parse_record_fields`, `parse_variant_constructors`, `extract_type_from_pattern`,
`extract_name_from_pattern`, `extract_param_from_pattern`, `parse_pattern`,
`parse_binop`, `parse_unop`, the `Ast_502`/`To_502`/`From_502` compat block +
`expression_to_502`/`expression_of_502`/`pattern_of_502`/`case_of_502`,
`type fun_body`, `is_function_expression_502`, `same_position`, `same_location`,
`expression_at_loc`, `pattern_of_param`, `collect_fun_params`, `is_function_expression`.
(`Parse_error_exn` MUST be here — the helpers raise it.)

**`Sarek_parse.ml`** (main) — keep lines ~262–end: the `let rec parse_let_shared …
and parse_superstep … and parse_expression … and is_array_access` chain,
`parse_kernel_function`, `parse_payload`. `open Sarek_parse_helpers`. Re-export the
external-API helpers:
```ocaml
exception Parse_error_exn = Sarek_parse_helpers.Parse_error_exn
let extract_name_from_pattern = Sarek_parse_helpers.extract_name_from_pattern
let extract_param_from_pattern = Sarek_parse_helpers.extract_param_from_pattern
let extract_type_from_pattern  = Sarek_parse_helpers.extract_type_from_pattern
let collect_fun_params         = Sarek_parse_helpers.collect_fun_params
let pattern_of_param           = Sarek_parse_helpers.pattern_of_param
```

## Quality Gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek/ppx/
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
ocamlformat --check sarek/ppx/Sarek_parse.ml sarek/ppx/Sarek_parse_helpers.ml
```
`@sarek/tests/runtest` builds the PPX and its consumers — the gate for the re-exports.

## Risks (6a)

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| External caller loses a helper | Med | Med | re-export the 6 listed names; runtest builds callers |
| Parse_error_exn identity split (two distinct exns) | Med | High | define ONCE in helpers; main re-exports via `exception … = …` (same identity) |
| Pure move alters logic | Low | High | verbatim move; runtest |
