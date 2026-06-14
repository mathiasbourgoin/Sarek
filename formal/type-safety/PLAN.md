# TypeSafety — Work Plan

**Last updated**: 2026-06-14 (scaffold — currentTask = T1-SPEC)
**Apparatus version**: 1.2.1 (inherited from convergence-safety template)
**Phase**: T1-SPEC (Formalise the Sarek type system in Rocq)
**Branch**: formal/type-safety-phase1a

---

## Targets (real source files)

| File | Lines | Role |
|---|---|---|
| `sarek/ppx/Sarek_typer.ml` | 1154 | Type inference engine (`infer : env -> expr -> (texpr * env) result`) |
| `sarek/ppx/Sarek_types.ml` | 343 | Type representation (`typ`) + unification (`unify`, `repr`, `occurs`) |
| `sarek/ppx/Sarek_typed_ast.ml` | 425 | Typed AST (`texpr`, `ty` field always resolved — no unbound TVar) |

The type system is GPU-aware Hindley-Milner with mutable unification variables.
Coq has no mutable unification, so the spec models **post-unification** types
(every type resolved, no TVar) — this mirrors `texpr.ty`.

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1-SPEC | Spec + basic lemmas: type universe, `type_env`/`lookup_env`, `infer_type`, `has_type`, 5 Admitted soundness lemmas | T1 | **current** | — |
| T1-SOUND | Prove the 5 lemmas; add type-preservation (subject reduction analogue) + completeness of `infer_type` vs `has_type` | T1 | todo | T1-SPEC |
| T1-CMBT | Extraction of `infer_type` to OCaml + differential conformance vs `Sarek_typer.infer` (CMBT closure) | T1 | todo | T1-SOUND |
| T2-CUSTOM | Extend type universe with TRecord/TVariant/Custom; field-access + constructor inference | T2 | TBD | T1-CMBT |
| T2-UNIFY | Model unification soundness (occurs check, `repr` idempotence, mgu) against `Sarek_types.unify` | T2 | TBD | T1-CMBT |
| T2-VEC | TVec / TArr memory-space inference (Local/Shared/Global) + vec/arr get/set typing | T2 | TBD | T1-CMBT |
| DOCS-SYNC | STATUS.md / FINDINGS.md / proof-ledger.json drift check | hygiene | clean (scaffold) | — |

---

## T1-SPEC acceptance (current task)

- [x] Type universe mirrors `Sarek_types.ml` (prim_type, reg_type, mem_space, sarek_type)
- [x] `type_env` + `lookup_env`
- [x] `infer_type` with literal/var/let rules matching `Sarek_typer.ml`
- [x] `has_type` declarative judgement (spec side)
- [x] 5 Admitted soundness lemmas stated
- [x] `theories/TypeSafetySpec.v` compiles (coqc / CoqMakefile, exit 0)
- [ ] Lemmas proved (→ T1-SOUND)

---

## Build

Coq theories are **not** built by dune (no `(using coq ...)` in the root
`dune-project`). They build via CoqMakefile, matching the convergence-safety
project:

```
cd formal/type-safety && rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile
```

The `formal/type-safety/extraction/` and `test/` dirs are reserved for the
T1-CMBT dune-driven OCaml extraction conformance harness (added in T1-CMBT).

---

## Next autopilot tick (T1-SPEC → T1-SOUND)

1. Prove `infer_lit_int`, `infer_lit_bool` — both by `reflexivity` / `simpl`.
2. Prove `infer_var_bound` and `lookup_env_sound` — induction on `env`, case on
   `String.eqb`.
3. Prove `infer_type_sound` — induction on `e`, using the lemmas above; the
   `ELet` case threads the binding through `has_type`'s `HT_Let`.
4. Once green, begin connecting to `Sarek_typer.infer` extraction (T1-CMBT).
