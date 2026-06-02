# Intake Brief — native-gen-split

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

`sarek/ppx/Sarek_native_gen.ml` is 1951 lines (3.9× the 500-line limit). Split it into
cohesive modules **by responsibility**, with zero change to generated OCaml output.

The file already uses an open-recursion design: the expression sub-generators
(`gen_memory_access`, `gen_let_binding`, `gen_control_flow`, `gen_data_structure`,
`gen_special_expr`, `gen_parallel_construct`) take the recursive `~gen_expr` as a
parameter rather than calling it directly. This makes them liftable into a separate
module without breaking the recursion.

**Value:** four files each closer to the size limit; clear separation between context,
expression generation, and kernel-level generation.

## Scope Boundary

OUT of scope — this is a **pure structural move**, not a behavior change:
- The confirmed semantic bugs documented in `kb/sarek/ppx/native-gen.md` MUST NOT be
  touched: `downto` bound reversal (lines ~305–343), `TECreateArray` int32/int
  mismatch (~466–469), custom-scalar `failwith` casting (~1157–1159), mutable inline-FCM
  rejection (~236–237). Each deserves its own intake. Moving them verbatim is required;
  fixing them is forbidden here.
- No change to `Sarek_native_helpers.ml` or `Sarek_native_intrinsics.ml`.
- No change to generated AST. Output must be byte-identical.
- No new functionality, no signature changes to the public entry points
  (`gen_expr`, `gen_expr_with_inline_types`, `gen_cpu_kern_native`,
  `gen_simple_cpu_kern_native`, `gen_cpu_kern_native_wrapper`, type-decl generators).

## Relevant Files

| File | Role |
|---|---|
| `sarek/ppx/Sarek_native_gen.ml` | The 1951-line monolith to split |
| `sarek/ppx/dune` | Library module list — must list the new modules |
| `sarek/ppx/Sarek_native_helpers.ml` | Dependency (unchanged) |
| `sarek/ppx/Sarek_native_intrinsics.ml` | Dependency (unchanged) |

## Architecture Notes

Proposed 4-module decomposition (names use the `Sarek_native_gen_*` prefix so they sit
in the existing `sarek_ppx_lib` library):

1. **`Sarek_native_gen_base.ml`** (current lines ~35–220) — leaf module, no internal deps:
   - `IntSet`, `StringSet` modules
   - `gen_context` type + `empty_ctx`
   - `is_same_module`, `types_module_var`
   - FCM name helpers (`field_getter_name`, `record_maker_name`, `variant_ctor_name`)
   - `gen_literal`, `gen_variable`
   - `custom_descriptor_expr`, inline-type helpers, `vector_type_id_expr`,
     `custom_type_id_expr`

2. **`Sarek_native_gen_expr.ml`** (current lines ~222–695) — depends on base:
   - the six `~gen_expr`-parameterised sub-generators

3. **`Sarek_native_gen.ml`** (remaining core) — depends on base + expr:
   - `gen_expr_impl` / `gen_pattern_impl` / `gen_binop` / `gen_unop` (the `let rec … and`)
   - entry points `gen_expr`, `gen_expr_with_inline_types`, `module_name_of_sarek_loc`
   - module item + type-decl generation, FCM module impl (`gen_module_fun`,
     `gen_type_decl_*`, `gen_module_impl`, `wrap_module_items`, `has_inline_types`, …)

4. **`Sarek_native_gen_kernel.ml`** (current lines ~1125–1728) — depends on base + expr + core:
   - `gen_mode_of_exec_strategy`, `gen_arg_cast`, `gen_types_object`,
     `gen_cpu_kern_native`, `gen_simple_cpu_kern_native`, `gen_cpu_kern_native_wrapper`

Dependency DAG (no cycles): base ← expr ← core ← kernel.

**Callers outside this file** reference these functions through the `Sarek_native_gen`
module name. After the split, any function that moved to `Sarek_native_gen_kernel`
(the public kernel builders) will have a new module path. Two options — the
implementer chooses the lower-churn one:
- (a) re-export from `Sarek_native_gen` via `let gen_cpu_kern_native = Sarek_native_gen_kernel.gen_cpu_kern_native` etc., OR
- (b) update the call sites.
Find callers first: `grep -rn 'Sarek_native_gen\.' sarek/ --include=*.ml`.

No `.mli` files currently exist for `Sarek_native_gen` (verify). If none exist, none
are required for the new modules either — but adding them is acceptable if it does not
expand scope.

## Quality Gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
ocamlformat --check sarek/ppx/Sarek_native_gen*.ml
./scripts/check-license-headers.sh
```

(Full `dune build` may fail on `-lnvrtc` — pre-existing CUDA-toolkit-absent issue,
unrelated. The `@sarek/tests/runtest` target is the real gate and includes native
codegen tests.)

## Open Questions

_(empty — design resolved by code reading)_
