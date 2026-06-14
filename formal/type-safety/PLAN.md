# TypeSafety — Work Plan

**Last updated**: 2026-06-14 (T3-S8 done, currentTask = T3-S9 TBD)
**Apparatus version**: 1.2.1 (inherited from convergence-safety template)
**Phase**: T3-S8 (done — 4 theorems, 12/12 smoke) → T3-S9 (TBD)
**Branch**: formal/convergence-safety-phase1a

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
| T1-SOUND | The 5 lemmas are proved; add type-preservation + completeness of `infer_type` vs `has_type` | T1 | **done** (9/9 Qed) | — |
| T1-CMBT | Extraction + differential conformance vs `Sarek_typer.infer` | T1 | **done** (2000/2000, found+fixed F-TS-01) | T1-SOUND |
| T2-CUSTOM | ETuple extension to TypeSafetySpec.v | T2 | **done** | T1-CMBT |
| T2-UNIFY | Unification soundness (occurs check, mgu) against `Sarek_types.unify` | T2 | **done** (1000/1000 differential, 0 admits) | T1-CMBT |
| T2-VEC | TVec/TArr inference + vec/arr get/set typing | T2 | **done** (5 theorems, 10/10 smoke) | T1-CMBT |
| T2-REGISTRY | TRecord/TVariant field access typing | T2 | **done** (5 theorems, 10/10 smoke) | T2-VEC |
| T3-S1 | Control flow (CFIfThen/IfElse/For/While/Seq) — ControlFlowSpec.v | T3 | **done** (4 theorems, 10/10 smoke) | T2-REGISTRY |
| T3-S2 | Operators (EBinop/EUnop) — OperatorSpec.v | T3 | **done** (4 theorems, 10/10 smoke) | T3-S1 |
| T3-S3 | Function application (EApp, ELetRec) — FunSpec.v | T3 | **done** (4 theorems, 10/10 smoke) | T3-S2 |
| T3-S4 | Mutable bindings (ELetMut, EAssign) — MutSpec.v | T3 | **done** (4 theorems, 10/10 smoke) | T3-S3 |
| T3-S5 | Pattern matching (EMatch) — PatternSpec.v | T3 | **done** (4 theorems + 2 aux, 11/11 smoke) | T3-S2 |
| T3-S6 | Algebraic construction (ERecord, EConstr) — ConstrSpec.v | T3 | **done** (4 theorems + 2 aux, 13/13 smoke) | T3-S5 |
| T3-S7 | Special forms (EReturn, ECreateArray, ETyped) — SpecialSpec.v | T3 | **done** (4 theorems, 12/12 smoke) | T3-S6 |
| T3-S8 | GPU forms (ELetShared, ESuperstep) — GPUSpec.v | T3 | **done** (4 theorems, 12/12 smoke) | T3-S7 |
| DOCS-SYNC | STATUS.md / FINDINGS.md / proof-ledger.json drift check | hygiene | clean | — |

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

## Next autopilot tick (T3-S8 — GPU forms)

T3-S7 is closed: 4 headline theorems (sound/complete/det/preservation) for
`SEReturn`/`SECreateArray`/`SETyped`, 12/12 smoke tests, 0 admits. Branch
formal/convergence-safety-phase1a. As-built modeling decisions (`SpecialSpec.v`):

- **SEReturn allowed body**: pass-through — `infer body = t; result = t`
  (Sarek_typer.ml:467). The `allowed : bool` AST flag makes `EarlyReturnNotAllowed`
  reachable: `allowed = true` is the faithful pass-through, `allowed = false`
  rejects. This keeps the `(env, mu, e)` inference signature identical to every
  prior layer rather than threading a tail-position context flag.
- **SECreateArray size elt mem**: `size` must equal `TPrim TInt32` (post-unification
  residue of Sarek's `unify_or_error tsize.ty t_int32`, line 470); result is
  `TArr elt mem` with the node's own memory space (not hard-coded Local). Error:
  `ArraySizeNotInt got`.
- **SETyped body annot**: `infer body = got`, assert `got = annot` (sarek_type_eq_dec);
  result is `annot` (Sarek line 505). Error: `TypeAnnotMismatch annot got`.

Error constructors: `SConstrErr` (delegated), `EarlyReturnNotAllowed`,
`ArraySizeNotInt`, `TypeAnnotMismatch`. No custom induction principle needed —
`special_expr` recurses only through single sub-expressions, so default
`induction e` yields the IHs.

T3-S8 adds `GPUSpec.v` for GPU forms (`ELetShared`, `ESuperstep`), wrapping the
special layer (`SpecialSpec.v`).

Spec file: `theories/GPUSpec.v`
Extraction: `extraction/GPUExtraction.v`
Model module: `GPUModel`

Divergence policy stays: any disagreement on a covered fragment is a model bug.
