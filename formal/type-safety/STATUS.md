# TypeSafety — Status

**Branch**: formal/type-safety-phase1a
**Phase**: T1-SPEC (started 2026-06-14)
**Toolchain**: Rocq 9.1.1 / OCaml 5.4.0

## Scoreboard

| Metric | Value |
|---|---|
| Theorems proven | 0 |
| Admits | 5 |
| Axioms | 0 |
| Definitions | infer_type, lookup_env, has_type, type universe |
| Build | green (coqc / CoqMakefile, exit 0) |

## Admitted (to prove in T1-SOUND)

1. `infer_lit_int` — `infer_type env (ELit (LInt n)) = inl (TPrim TInt32)`
2. `infer_lit_bool` — `infer_type env (ELit (LBool b)) = inl (TPrim TBool)`
3. `infer_var_bound` — `lookup_env env x = Some t -> infer_type env (EVar x) = inl t`
4. `lookup_env_sound` — `lookup_env env x = Some t -> In (x, t) env`
5. `infer_type_sound` — `infer_type env e = inl t -> has_type env e t`

## Notes

- Spec models **post-unification** types (no TVar) — mirrors `texpr.ty`
  ("Always resolved, never contains unbound TVar", Sarek_typed_ast.ml:29).
- Coq theories build via CoqMakefile, not dune (root `dune-project` has no
  `(using coq ...)`; following the convergence-safety pattern).
