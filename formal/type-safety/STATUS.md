# TypeSafety — Status

**Branch**: formal/type-safety-phase1a
**Phase**: T1-SOUND (done 2026-06-14) → T1-CMBT (current)
**Toolchain**: Rocq 9.1.1 / OCaml 5.4.0

## Scoreboard

| Metric | Value |
|---|---|
| Theorems proven | 9 |
| Admits | 0 |
| Axioms | 0 |
| Definitions | infer_type, lookup_env, has_type, type universe |
| Build | green (coqc / CoqMakefile, exit 0; dune build/test, exit 0) |
| Conformance harness | scaffold green (13/13 smoke tests) |

## Proven (tick 1, all Qed)

1. `infer_lit_int` — `infer_type env (ELit (LInt n)) = inl (TPrim TInt32)` (reflexivity)
2. `infer_lit_bool` — `infer_type env (ELit (LBool b)) = inl (TPrim TBool)` (reflexivity)
3. `infer_var_bound` — `lookup_env env x = Some t -> infer_type env (EVar x) = inl t` (intros; rewrite H; reflexivity)
4. `lookup_env_sound` — `lookup_env env x = Some t -> In (x, t) env` (induction on env, t generalized; String.eqb_eq)
5. `infer_type_sound` — `infer_type env e = inl t -> has_type env e t` (structural induction on e, env reverted)

## Proven (tick 2 — T1-SOUND, all Qed)

6. `lookup_env_correct` — `lookup_env` is a partial function (at most one result); rewrite + injection
7. `has_type_det` — determinism of `has_type`; induction on first derivation, inversion on second (EVar via `lookup_env_correct`, ELet chains both IHs)
8. `infer_type_complete` — `has_type env e t -> infer_type env e = inl t` (converse of `infer_type_sound`); induction on `has_type`
9. `type_preservation` (main) — `infer_type env e = inl t <-> has_type env e t`; split into sound + complete

## T1-CMBT scaffold (tick 2)

- `extraction/TypeSafetyExtraction.v` — Rocq extraction config; extracts `infer_type`, `lookup_env` to `TypeSafetyModel.ml`
- `extraction/dune` — `type_safety_model` library stanza (`-w -a` on generated code)
- `test/test_type_safety_conformance.ml` — 13 smoke tests (literals, bound/unbound var, let simple/shadow/error, lookup_env hit/miss/first-wins)
- `test/dune` — test stanza depending on `type_safety_model`
- Status: `dune build` exit 0, `dune test` 13/13 passed, exit 0. Differential conformance vs `Sarek_typer.infer` is the next T1-CMBT step (not yet wired).

## Notes

- Spec models **post-unification** types (no TVar) — mirrors `texpr.ty`
  ("Always resolved, never contains unbound TVar", Sarek_typed_ast.ml:29).
- Coq theories build via CoqMakefile, not dune (root `dune-project` has no
  `(using coq ...)`; following the convergence-safety pattern).
- Extraction emits its own `string`/`String` constructors that shadow OCaml's
  `String` module — `coq_string_of_string` must be defined before `open M`.
