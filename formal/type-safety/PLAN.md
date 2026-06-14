# TypeSafety — Work Plan

**Last updated**: 2026-06-14 (tick 3 — T1-CMBT done, currentTask = T2)
**Apparatus version**: 1.2.1 (inherited from convergence-safety template)
**Phase**: T1-CMBT (done — differential 2000/2000, found+fixed F-TS-01) → T2
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
| T1-SPEC | Spec + basic lemmas: type universe, `type_env`/`lookup_env`, `infer_type`, `has_type`, 5 soundness lemmas | T1 | **done** (5/5 Qed) | — |
| T1-SOUND | The 5 lemmas are proved; add type-preservation (subject reduction analogue) + completeness of `infer_type` vs `has_type` | T1 | **done** (4 new: lookup_env_correct, has_type_det, infer_type_complete, type_preservation; 9/9 Qed) | — |
| T1-CMBT | Extraction of `infer_type` to OCaml + differential conformance vs `Sarek_typer.infer` (CMBT closure) | T1 | **done** (differential 2000/2000, 0 errors/fails; found+fixed F-TS-01 ELet scope leak) | T1-SOUND |
| T2-CUSTOM | Extend type universe with TRecord/TVariant/Custom; field-access + constructor inference | T2 | **next** | T1-CMBT |
| T2-UNIFY | Model unification soundness (occurs check, `repr` idempotence, mgu) against `Sarek_types.unify` | T2 | TBD | T1-CMBT |
| T2-VEC | TVec / TArr memory-space inference (Local/Shared/Global) + vec/arr get/set typing | T2 | TBD | T1-CMBT |
| DOCS-SYNC | STATUS.md / FINDINGS.md / proof-ledger.json drift check | hygiene | clean (scaffold) | — |

---

## T1-SPEC acceptance (current task)

- [x] Type universe mirrors `Sarek_types.ml` (prim_type, reg_type, mem_space, sarek_type)
- [x] `type_env` + `lookup_env`
- [x] `infer_type` with literal/var/let rules matching `Sarek_typer.ml`
- [x] `has_type` declarative judgement (spec side)
- [x] 5 soundness lemmas stated
- [x] `theories/TypeSafetySpec.v` compiles (coqc / CoqMakefile, exit 0)
- [x] Lemmas proved (5/5 Qed, tick 1)

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

## Next autopilot tick (T2 — extend the type universe)

T1-CMBT is closed: the differential of the extracted `infer_type` against the
real `Sarek_typer.infer` is green (2000/2000, 0 errors/fails) and surfaced +
fixed F-TS-01 (ELet scope leak). The model covers the literal/var/let fragment.

T2 widens the modelled fragment. Recommended order:

1. **T2-CUSTOM** (next) — extend the type universe and `infer_type`/`has_type`
   with TRecord/TVariant/Custom, field-access and constructor inference; re-run
   the existing differential plus new generators over the widened fragment.
2. **T2-UNIFY** — model unification soundness (occurs check, `repr` idempotence,
   mgu) against `Sarek_types.unify`; this is where the model goes from
   post-unification to modelling the solver itself.
3. **T2-VEC** — TVec/TArr memory-space inference (Local/Shared/Global) + vec/arr
   get/set typing.

Divergence policy stays: any disagreement on a covered fragment is a model bug
(the model is the spec) — record in FINDINGS.md, do not silently widen the model.
