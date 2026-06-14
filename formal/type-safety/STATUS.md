# TypeSafety — Status

**Branch**: formal/convergence-safety-phase1a
**Phase**: T3-S8 (GPUSpec, done 2026-06-14) -> T3-S9 (next)
**Toolchain**: Rocq 9.1.1 / OCaml 5.4.0

## Scoreboard

| Metric | Value |
|---|---|
| Theorems proven | 90 (78 headline + 12 auxiliary, all Qed/Defined) |
| Admits | 0 |
| Axioms | 0 |
| Definitions | infer_type, lookup_env, has_type, pre_type, follow, follow_pvar, occurs_in, unify_fun, apply_subst, infer_mem_type, sarek_type_eq_dec, has_mem_type, field_lookup, infer_rec_type, has_rec_type, infer_cf_type, has_cf_type, is_numeric, is_integer, infer_op_type, has_op_type, infer_fun_type, has_fun_type, is_mutable, infer_mut_type, has_mut_type, lookup_constr, branch_body_env, check_branches, infer_pat_type, has_pat_type, branches_have_type, check_fields, infer_constr_type, has_constr_type, fields_have_type, infer_special_type, has_special_type, infer_gpu_type, has_gpu_type |
| Build | green (CoqMakefile, exit 0; dune build/test, exit 0) |
| T1-CMBT harness | green -- differential QCheck 2000/2000 (0 errors, 0 fails) + 20/20 smoke |
| T2-UNIFY harness | green -- differential QCheck 1000/1000 (0 errors, 0 fails) + 15/15 smoke |
| T2-VEC harness | green -- 10/10 smoke (EVecGet/EVecSet/EArrGet/EArrSet + error cases) |
| T2-REGISTRY harness | green -- 10/10 smoke (EFieldGet/EFieldSet + error cases + nested + vec delegation) |
| T3-S1 harness | green -- 10/10 smoke (CFIfThen/IfElse/For/While/Seq + error cases + loop var scope) |
| T3-S2 harness | green -- 10/10 smoke (Add/Mod/Eq/Lt/And/Neg/Not/Lnot + mismatch + NotBool) |
| T3-S3 harness | green -- 10/10 smoke (FEOp delegation/binop, FEApp success, NotAFunc, ArgMismatch, FELetRec success/recursive/param-scope, BodyMismatch, param-not-leaked) |
| T3-S4 harness | green -- 10/10 smoke (MEFun delegation, MELetMut success/body-type/nested, MEAssign success, MEUnbound, MEImmutable, MEAssignMismatch, MEFunErr propagation, init-error short-circuit) |
| T3-S5 harness | green -- 11/11 smoke (PEMut delegation, no-payload match, payload-bound match, PENotVariant, PEMismatch unknown constructor, PEBranchType, PEMutErr delegation, PEEmpty, scoped binder, multi-branch agree, error short-circuit) |
| T3-S6 harness | green -- 13/13 smoke (CEPat delegation, record success/empty-provided, FieldTypeMismatch, UnknownField, nullary constr, payload constr, payload mismatch, UnknownConstr, ConstrArity extra-arg/missing-arg, CPatternErr delegation, nested construction) |
| T3-S7 harness | green -- 12/12 smoke (SEConstr delegation, return pass-through int32/bool, EarlyReturnNotAllowed, create_array Global/Shared, ArraySizeNotInt, type-annot match/mismatch, SConstrErr delegation, nested return/typed/array, size-error propagation through SETyped) |
| T3-S8 harness | green -- 12/12 smoke (GESpecial delegation + GSpecialErr, let%shared success/body-type, SharedNotArray, SharedNotShared Global/Local, superstep success/SuperstepBodyNotUnit, let%shared env scoping, nested superstep+let%shared, error propagation through superstep) |
| Findings | F-TS-01 (ELet scope leak) -- found by T1-CMBT, RESOLVED |

## Termination design (T2-UNIFY)

The key challenge in formalizing the HM unifier was Rocq's guard checker.
Three design decisions make it work:

1. **follow as Definition**: `follow` dispatches on `pre_type` with a match.
   Ground constructors (PPrim/PReg/PTuple) return the argument immediately.
   PVar delegates to `follow_pvar` (a Fixpoint on fuel). Since `follow` is a
   Definition (not a Fixpoint), `follow fuel s (PPrim p)` reduces by iota
   regardless of the fuel value.

2. **{struct fuel} annotation**: Without it, Rocq infers `pre_subst` (a list)
   as the structural argument of `unify_fun`, causing the recursive call to
   be rejected. The annotation forces `fuel : nat` as the decreasing argument.

3. **unify_fun n (not S n) for PTuple**: The tuple case uses `unify_fun n`
   (the predecessor), which is strictly smaller than the outer fuel `S n`.
   This satisfies the guard checker.

## Proof technique notes (T2-VEC)

1. **sarek_type_eq_dec**: Uses `fix IH 1` with explicit `destruct t1 as [...];
   destruct t2 as [...]` patterns to avoid Rocq auto-naming conflicts. Inner
   IHL fix handles list equality for TFun/TTuple cases.

2. **infer_mem_type_sound**: Uses `intros env e t H; revert env t H; induction e`
   to avoid the `env already used` error that `induction e; intros env t H`
   would cause (induction on a forall-quantified variable auto-introduces
   leading quantifiers before the induction variable).

3. **infer_mem_type_complete**: Uses `intros env0 e0 t0 H; induction H` with
   `env0` to avoid clash with the `env` introduced by the HMT_Core constructor's
   forall. Uses `match goal with Hht : has_type _ _ _ |- _ =>` to find the
   has_type hypothesis without depending on its auto-generated name.

4. **has_mem_type_det**: Piggybacks on completeness -- infer_mem_type is a
   function, so two outputs for the same input must agree.

## Proven (tick 1-2, TypeSafetySpec.v -- T1)

(see proof-ledger.json for full list of 9 T1 theorems)

## Proven (tick 3, UnifySpec.v -- T2-UNIFY, all Qed, 0 admits)

21. `pre_type_ind_strong` -- custom induction for pre_type (Forall IH for PTuple)
22. `follow_ground_prim/reg/tuple` -- ground types are fixed by follow (3 lemmas)
23. `follow_pvar_none` -- unbound PVar follows to itself
24. `prim_type_beq_true/false` -- prim_type_beq reflects =/!= (2 lemmas)
25. `reg_type_beq_true/false` -- reg_type_beq reflects =/!= (2 lemmas)
26. `prim_type_beq_refl` -- prim_type_beq p p = true
27. `reg_type_beq_refl` -- reg_type_beq r r = true
28. `subst_lookup_head` -- head binding lookup
29. `occurs_ground_false` -- PPrim/PReg contain no PVar (2 lemmas)
30. `unify_fun_prim_prim/reg_reg` -- unfolding lemmas (2 lemmas)
31. `unify_fun_var_prim/var_reg` -- conditional unfolding for PVar/ground (2 lemmas)
32. `unify_zero_none` -- zero fuel always returns None
33. `unify_prim_sound/complete` -- PPrim soundness and completeness
34. `unify_reg_sound/complete` -- PReg soundness and completeness
35. `unify_var_binds_prim/reg` -- free PVar gets bound to ground type
36. `occurs_check_blocks_prim/reg` -- occurs_in ground always false

## Proven (tick 4, VecSpec.v -- T2-VEC, all Qed, 0 admits)

37. `sarek_type_eq_dec` -- decidable equality on all sarek_type constructors
38. `infer_mem_type_sound` -- EVecGet/EVecSet/EArrGet/EArrSet inference -> has_mem_type
39. `infer_mem_type_complete` -- has_mem_type -> EVecGet/EVecSet/EArrGet/EArrSet inference
40. `has_mem_type_det` -- uniqueness of the declarative mem_expr judgement
41. `mem_type_preservation` -- infer_mem_type env e = inl t <-> has_mem_type env e t

## Proven (tick 5, RegistrySpec.v -- T2-REGISTRY, all Qed, 0 admits)

42. `field_lookup_sound` -- successful lookup implies membership in field list
43. `infer_rec_type_sound` -- EFieldGet/EFieldSet inference -> has_rec_type
44. `infer_rec_type_complete` -- has_rec_type -> EFieldGet/EFieldSet inference
45. `has_rec_type_det` -- uniqueness of the declarative rec_expr judgement
46. `rec_type_preservation` -- infer_rec_type env e = inl t <-> has_rec_type env e t

## Sarek_type extensions (T2-REGISTRY)

TRecord and TVariant are now full constructors in `sarek_type`:
- `TRecord : string -> list (string * sarek_type) -> sarek_type`
- `TVariant : string -> list (string * option sarek_type) -> sarek_type`

`sarek_type_eq_dec` extended with cases for both new constructors (uses nested
`fix IHLR/IHLV` for decidability of `list (string * sarek_type)` and
`list (string * option sarek_type)` respectively).

## Proof technique notes (T2-REGISTRY)

1. **sarek_type_eq_dec TRecord/TVariant**: Uses nested `fix IHLR 1` / `fix IHLV 1`
   inside the main `fix IH 1` proof to establish decidability of the field/constructor
   lists. Key insight: Rocq 9's `injection` on `(c1, Some t1) :: rest1 = (c2, Some t2) :: rest2`
   already decomposes the pair and option, yielding `Hc : c1 = c2`, `Hot : t1 = t2`,
   `Hrest : rest1 = rest2` — three names, not four. For `None/None` pairs, only
   two names (no option injection produced for trivially equal None values).

2. **infer_rec_type_sound EFieldGet/EFieldSet**: Destructs on 8-constructor `sarek_type`
   with `try discriminate` closing all non-TRecord cases; then applies IH + HRT_FieldGet/Set.

3. **infer_rec_type_complete**: Follows VecSpec pattern — `match goal with Hmem` to
   locate has_mem_type hypothesis; HRT_FieldSet uses `sarek_type_eq_dec field_t field_t`
   left branch for reflexivity.

## Proven (tick 6, ControlFlowSpec.v -- T3-S1, all Qed, 0 admits)

47. `infer_cf_type_sound` -- CFIfThen/IfElse/For/While/Seq inference -> has_cf_type
48. `infer_cf_type_complete` -- has_cf_type -> inference
49. `has_cf_type_det` -- uniqueness of the declarative cf_expr judgement
50. `cf_type_preservation` -- infer_cf_type env e = inl t <-> has_cf_type env e t

## Proof technique notes (T3-S1)

1. **Two if-constructors**: `CFIfThen` (no else, then must be unit) and `CFIfElse`
   (branches must agree) instead of `CFIf ... (option cf_expr)`. Avoids option
   inside the inductive, so Rocq's default induction gives IHs for all direct
   subterms without a custom induction principle.

2. **sarek_type_eq_dec self-check in completeness**: For branches like
   `sarek_type_eq_dec (TPrim TBool) (TPrim TBool)`, `destruct` gives left/right.
   The `right` branch is dismissed with `exfalso; apply Hne; reflexivity`.
   This is the standard pattern for eq_dec on provably-equal arguments.

3. **CFFor loop variable**: The bound variable `var` is added to the environment
   as `(var, TPrim TInt32)` before inferring the body. The body type is ignored
   (result is always unit); only the body's *successful* inference matters.
   The IH for `body` applies with the extended environment directly.

4. **has_cf_type_det via piggybacking**: Same technique as RegistrySpec.v —
   `infer_cf_type_complete H1/H2` gives two equations; rewriting one into the
   other and injecting gives the equality. No case analysis on `e` needed.

## Proven (tick 7, OperatorSpec.v -- T3-S2, all Qed, 0 admits)

51. `infer_op_type_sound` -- OPBinop/OPUnop inference -> has_op_type
52. `infer_op_type_complete` -- has_op_type -> inference
53. `has_op_type_det` -- uniqueness of the declarative op_expr judgement
54. `op_type_preservation` -- infer_op_type env e = inl t <-> has_op_type env e t

## Proof technique notes (T3-S2)

1. **19-goal OPBinop dispatch**: `destruct op; simpl in H` opens 19 subgoals (one per
   binop constructor). The `all: try (...)` pattern is UNRELIABLE in Rocq 9 Ltac1 —
   partial success leaves goals in an inconsistent state. Use explicit `N-M:` goal
   selectors throughout: `1-4:` (Add/Sub/Mul/Div numeric), `1:` (Mod integer), `1-2:`
   (And/Or bool), `1-2:` (Eq/Ne eq), `1-4:` (Lt/Le/Gt/Ge cmp), `1-6:` (Land...Asr integer).

2. **Iota substitution eliminates rewrite**: `destruct (sarek_type_eq_dec t1 t2) as [Hb|Hne]`
   automatically reduces `match sarek_type_eq_dec t1 t2 with Left _ => ... | Right _ => ...`
   in the context via iota substitution. No `rewrite` or `simpl` needed after destruct.
   Adding `eqn:Heqb` followed by `rewrite Heqb in H` FAILS with "term not found in H".

3. **HOT_BoolBinop rewrite pattern**: For And/Or, the spec rewrites the bool constraint
   into the IH hypotheses before applying the constructor. Specifically, `Hb : t1 = TPrim TBool`
   is used as `rewrite Hb in Hl; rewrite Hb in Hr` (not `subst`) to avoid "variable used
   in conclusion" errors when `t1` still appears in the goal.

4. **Completeness hypothesis naming**: `induction H; simpl.` for constructor
   `HOT_NumericBinop env op lhs rhs t (P1) (IH1) (IH2) (P2)` gives H=P1 (first
   non-recursive premise), IHhas_op_type1=IH1, IHhas_op_type2=IH2, H0=P2 (second
   non-recursive). Use `destruct (is_numeric t) eqn:Hnum; [reflexivity | congruence]`
   to avoid relying on these names — `congruence` finds the contradiction between
   `Hnum : is_numeric t = false` and the constructor's `is_numeric t = true` premise.

## Proven (tick 8, FunSpec.v -- T3-S3, all Qed, 0 admits)

55. `infer_fun_type_sound` -- FEApp/FELetRec inference -> has_fun_type
56. `infer_fun_type_complete` -- has_fun_type -> inference
57. `has_fun_type_det` -- uniqueness of the declarative fun_expr judgement
58. `fun_type_preservation` -- infer_fun_type env e = inl t <-> has_fun_type env e t

## Proof technique notes (T3-S3)

1. **Single-param function model**: `TFun (list sarek_type) sarek_type` is constrained to a
   one-element parameter list `p_ty :: nil` in both the algorithm and the judgement, matching
   the EApp regular-application branch of Sarek_typer.ml in the post-unification model.

2. **TFun destruct cascade in soundness**: After `destruct (infer_fun_type env fn)` to `tfn`,
   peel the function type with `destruct tfn as [...|...|...] ; try discriminate` (8 sarek_type
   constructors), then `destruct params as [|p_ty rest]; try discriminate` and
   `destruct rest as [|]; try discriminate` to isolate the single-param `[p_ty]` shape. Every
   non-`TFun [p_ty] ret` shape falls into the `NotAFunc` inr branch and is killed by `discriminate`.

3. **No mutual recursion needed**: `fun_expr` recurses only through `fun_expr` sub-terms
   (FEApp fn/arg, FELetRec body/cont), so the default `induction e` principle yields all four
   IHs directly — no `expr_ind_strong`-style custom scheme required.

4. **Env extension threading**: FELetRec inference extends the body env with
   `(p_name,p_ty)::(fn_name,fn_ty)::env` and the continuation env with `(fn_name,fn_ty)::env`.
   The judgement `HFT_LetRec` mirrors these exact env shapes, so soundness/completeness are
   direct constructor applications with the matching IH on each sub-derivation.

## Proven (tick 9, MutSpec.v -- T3-S4, all Qed, 0 admits)

59. `infer_mut_type_sound` -- MELetMut/MEAssign inference -> has_mut_type
60. `infer_mut_type_complete` -- has_mut_type -> inference
61. `has_mut_type_det` -- uniqueness of the declarative mut_expr judgement
62. `mut_type_preservation` -- infer_mut_type env mu e = inl t <-> has_mut_type env mu e t

## Proof technique notes (T3-S4)

1. **Dual environment**: mutability is tracked by a second environment `mut_env := list string`
   (names currently bound mutable), threaded alongside the ordinary `type_env`. `MELetMut`
   adds `(name,t)` to the `type_env` and `name` to the `mut_env`; `MEFun` delegation reads only
   the `type_env`, so a mutable variable can still be read through the function layer.

2. **EAssign rules mirror Sarek_typer.ml infer_let_binding**: lookup `name`; `None -> MEUnbound`;
   bound but not in `mut_env` (i.e. `vi_mutable` false / `vi_is_param`) `-> MEImmutable`; value
   type `<>` declared type `-> MEAssignMismatch`; otherwise the result type is `TPrim TUnit`.

3. **Soundness ordering**: `induction e` is used with `intros env mu t H` *after* the induction
   so each IH is universally quantified over `env`/`mu` (the body env differs from the outer env).
   The `MEAssign` case destructs value result, `lookup_env`, `is_mutable` (eqn), and
   `sarek_type_eq_dec tv tdecl`, then applies `HMT_Assign` with the explicit witnesses.

4. **Completeness for HMT_Assign**: `subst tv` collapses the `tv = tdecl` premise so the
   `sarek_type_eq_dec tdecl tdecl` match reduces to the `left` branch via the
   `[reflexivity | exfalso; apply Hne; reflexivity]` idiom; the `lookup`/`is_mutable` hypotheses
   are rewritten directly.

## Proven (tick 10, PatternSpec.v -- T3-S5, all Qed, 0 admits)

63. `infer_pat_type_sound` -- PEMatch inference -> has_pat_type
64. `infer_pat_type_complete` -- has_pat_type -> inference
65. `has_pat_type_det` -- uniqueness of the declarative pat_expr judgement
66. `pat_type_preservation` -- infer_pat_type env mu e = inl t <-> has_pat_type env mu e t

## Proof technique notes (T3-S5)

1. **Single-level variant patterns**: each branch is `(constructor_name, opt bound_var, body)`.
   The scrutinee must infer to `TVariant _ constrs`; `lookup_constr` finds the constructor's
   optional payload type; `branch_body_env` prepends `(v, payload)` to the env for the body
   only when the constructor carries a payload and the branch binds it (mirrors `add_var`
   in Sarek_typer.ml infer_pattern; pattern binders are immutable, so the `mut_env` is left
   unchanged). The first branch fixes `result_ty`; every subsequent branch must agree.

2. **`branch_body_env` shared by Fixpoint and judgement**: the inference `Fixpoint` and the
   declarative `Inductive` both call the same `branch_body_env`; this is load-bearing — when a
   `destruct (infer_pat_type (branch_body_env ...) ...)` is performed, the term must appear
   *syntactically* in the hypothesis being case-split, otherwise `discriminate` fails with
   "Not a discriminable equality". An earlier version inlined the env match in the Fixpoint
   and the `destruct` term diverged from the hypothesis.

3. **Strong induction binder order**: `pat_expr_ind_strong` places `P scrut` *before* the
   `branches` binder: `(forall scrut, P scrut -> forall branches, Forall (..) branches -> ..)`.
   With the naive `(forall scrut branches, P scrut -> Forall .. -> ..)` order, the pattern
   `as [me | scrut IHscrut branches Hbranches]` binds `IHscrut` to the *branches list* (off
   by one), which silently breaks every later tactic.

4. **`inversion Hbranches`, not `destruct branches`**: in the soundness `PEMatch` case,
   `destruct branches` fails with "Unable to find an instance for the variables env, mu, t":
   the strong-induction predicate quantifies over `env`/`mu`/`t`, so the elimination motive
   cannot be inferred when those quantifiers sit inside the `Forall`. Case-splitting via
   `inversion Hbranches` (the `Forall` over branch bodies) sidesteps the motive inference and
   hands back the head IH (`Hhead`) and tail Forall (`Htail`) directly.

5. **Head/rest split aligns judgement with Fixpoint**: `HPT_Match` makes the head branch
   explicit and recurses with `branches_have_type` over the *rest*, exactly as `infer_pat_type`
   peels the first branch (to fix `result_ty`) then runs `check_branches` over the rest. This
   one-to-one structural alignment makes completeness a straight `rewrite` chain over the three
   IHs (scrut, head body, rest-check) produced by `has_pat_type_mind`.

## Proven (tick 11, ConstrSpec.v -- T3-S6, all Qed, 0 admits)

67. `infer_constr_type_sound` -- CERecord/CEConstr inference -> has_constr_type
68. `infer_constr_type_complete` -- has_constr_type -> inference
69. `has_constr_type_det` -- uniqueness of the declarative constr_expr judgement
70. `constr_type_preservation` -- infer_constr_type env mu e = inl t <-> has_constr_type env mu e t

## Proof technique notes (T3-S6)

1. **Construction vs. matching duality**: PatternSpec *destructs* algebraic values;
   ConstrSpec *builds* them. `CERecord rname declared provided` carries the declared
   record layout (the `TRecord` it targets) plus the provided `(field, value)` list;
   each provided field's inferred type must equal the *declared* type of the field of
   that name (reusing `RegistrySpec.field_lookup` over `declared`). The result is the
   declared `TRecord rname declared`. `CEConstr tyname constrs cname arg` carries the
   full variant constructor list; `lookup_constr` (reused from PatternSpec) finds the
   chosen constructor's optional payload; the result is the full `TVariant tyname constrs`,
   matching Sarek's `full_variant_ty`.

2. **Four-way payload x arg split for CEConstr** mirrors Sarek's `Wrong_arity` cases:
   `(None,None)` -> ok (HCT_ConstrNone); `(Some pty, Some a)` -> infer `a`, check
   `sarek_type_eq_dec got pty` (HCT_ConstrSome / FieldTypeMismatch); the two mixed cases
   `(Some,_None)` and `(None, Some)` -> `ConstrArity`. In soundness these mixed cases close
   by `discriminate` since the Fixpoint returns `inr`; only the two matching arities yield `inl`.

3. **Explicit `with (pty := pty)` for HCT_ConstrSome**: the payload type `pty` appears only
   in the constructor's *premises* (`lookup_constr ... = Some (Some pty)` and the recursive
   `has_constr_type arg pty`), never in its conclusion `TVariant tyname constrs`. `apply
   HCT_ConstrSome` therefore cannot unify `pty` and fails with "Unable to find an instance
   for the variable pty"; supplying it explicitly resolves the elimination.

4. **`check_fields` shares the declared layout with the judgement**: as in PatternSpec, the
   inference `Fixpoint` (`check_fields`) and the declarative `Inductive` (`fields_have_type`)
   both call `field_lookup fname declared` and `sarek_type_eq_dec`, keeping the case-split
   terms syntactically identical so `discriminate`/`injection` line up. Soundness over the
   provided list is `check_fields_sound`, fed the per-field `Forall` from the strong induction.

5. **Strong-induction payload arm uses a `match`-shaped IH**: `constr_expr_ind_strong` gives the
   CEConstr arm the hypothesis `match arg with Some a => P a | None => True end` rather than a
   bare `P arg`, because the payload is optional; the `None` case carries the trivial `True`.
   This is the optional-field analogue of PatternSpec's `Forall`-over-branches IH.

## Proven (tick 12, SpecialSpec.v -- T3-S7, all Qed, 0 admits)

71. `infer_special_type_sound` -- SEReturn/SECreateArray/SETyped inference -> has_special_type
72. `infer_special_type_complete` -- has_special_type -> inference
73. `has_special_type_det` -- uniqueness of the declarative special_expr judgement
74. `special_type_preservation` -- infer_special_type env mu e = inl t <-> has_special_type env mu e t

## Proof technique notes (T3-S7)

1. **Three special forms mirror `infer_data_structure`/`infer_special`** (Sarek_typer.ml:467/470/505):
   `SEReturn allowed body` is a *pass-through* -- its type is exactly the body's type, matching
   `EReturn e -> mk_texpr (TEReturn te) te.ty`. `SECreateArray size elt mem` infers `size`,
   requires it to be `TPrim TInt32` (the post-unification residue of Sarek's
   `unify_or_error tsize.ty t_int32`), and returns `TArr elt mem` exactly as Sarek's `arr_ty`.
   `SETyped body annot` infers `body`, requires `got = annot` (residue of `unify_or_error te.ty ty`),
   and returns the annotation type (Sarek: `{te with ty = repr ty}`).

2. **`EarlyReturnNotAllowed` reachability via an explicit `allowed : bool` AST flag**: Sarek itself
   never rejects a return, but a return is only meaningful in a tail position. Rather than thread a
   context flag through `(env, mu)` -- which would diverge from the established layer signature -- the
   side condition is carried by the `SEReturn` node. `allowed = true` is the faithful pass-through;
   `allowed = false` yields `EarlyReturnNotAllowed`, keeping the error constructor genuinely reachable
   (smoke test 4) without perturbing the inference signature shared by all layers.

3. **No custom induction principle needed**: unlike the list-bearing layers (ETuple/branches/fields),
   `special_expr` recurses only through single sub-expressions (`SEReturn`/`SECreateArray`/`SETyped`
   each hold one `special_expr`), so Rocq's default `induction e` already yields the needed IHs.
   Soundness is a flat 4-case induction; the `false` return arm closes by `discriminate`.

4. **`subst sz` / `subst gt` after `sarek_type_eq_dec`**: in soundness the `left` branch of the
   decidable equality gives `Heq : sz = TPrim TInt32` (resp. `got = annot`); `subst` rewrites the
   inference hypothesis so the recursive IH (`IHsize`/`IHbody`) applies at the expected type. This is
   the same reflexive-`eq_dec` collapse used in completeness, where `destruct (sarek_type_eq_dec X X)`
   discharges the `right` (Hne) arm by `exfalso; apply Hne; reflexivity`.

## Proven (tick 13, GPUSpec.v -- T3-S8, all Qed, 0 admits)

75. `infer_gpu_type_sound` -- GESpecial/GELetShared/GESuperstep inference -> has_gpu_type
76. `infer_gpu_type_complete` -- has_gpu_type -> inference
77. `has_gpu_type_det` -- uniqueness of the declarative gpu_expr judgement
78. `gpu_type_preservation` -- infer_gpu_type env mu e = inl t <-> has_gpu_type env mu e t

## Proof technique notes (T3-S8)

1. **Two BSP/GPU forms mirror `infer`'s `ELetShared`/`ESuperstep` arms** (Sarek_typer.ml:769/800):
   `GELetShared name ty body` models Sarek's `let%shared` -- Sarek binds `name` to
   `TArr (elem_t, Shared)` (line 781) and infers the body, returning `tbody.ty`. In this
   post-unification model the AST node carries the fully-resolved array type `ty` directly; the
   load-bearing well-formedness condition is that `ty` is a *shared array* (shape `TArr _ Shared`).
   `GESuperstep body` models `let%superstep` -- Sarek unifies the step body's type with `t_unit`
   (line 803); we require `body : TPrim TUnit` and the construct itself yields `TPrim TUnit`
   (a barrier-delimited side-effecting block).

2. **Two distinct error constructors from one `match ty`**: the inference splits `ty` once --
   `TArr elt Shared` recurses, `TArr _ _` (any non-Shared space) yields `SharedNotShared`, and every
   non-array shape falls through to `SharedNotArray`. This keeps both error arms genuinely reachable
   (smoke tests 5/6/7 cover non-array, Global, and Local respectively) while the success arm is the
   only path that extends the environment.

3. **Env scoping via `(name, ty) :: env`**: `GELetShared` binds the shared-array name into the
   `type_env` before inferring the body, exactly as the lower layers thread let-bindings. The
   declarative `HGT_LetShared` constructor pins the bound type to `TArr elt Shared`, so completeness
   needs no `sarek_type_eq_dec` on the binding -- the inference's `match ty` already computes to the
   recursive call under the extended env (`simpl; exact IHbody`). Smoke test 10 verifies the binding
   is precise (the bound name resolves; a sibling name still fails as `GSpecialErr`).

4. **No custom induction principle needed**: neither `GELetShared` nor `GESuperstep` holds a *list*
   of sub-expressions (each recurses through a single `gpu_expr`), so Rocq's default `induction e`
   yields the required IHs directly. Soundness is a flat 3-case induction: the `GELetShared` arm
   `destruct ty` (only `TArr` survives `discriminate`), then `destruct mem` (only `Shared` survives),
   reducing to the body IH; the `GESuperstep` arm uses the reflexive-`sarek_type_eq_dec` collapse
   (with `subst bt`) shared by the special/operator layers.

## Auxiliary lemmas (scaffolding, all Qed/Defined, 0 admits)

These 12 declarations are internal scaffolding for the headline theorems above
(ETuple/list recursion + the three custom induction principles + the branch-list and
field-list helpers). They are not headline results but are tracked in the proof ledger so the
declaration count is exact (78 headline + 12 auxiliary = 90 total
`Theorem`/`Lemma` declarations).

63. `expr_ind_strong` -- custom strong induction principle for `expr` (Forall IH for ETuple); `Defined`
64. `infer_type_etuple` -- unfolds the ETuple case of `infer_type` (reflexivity)
65. `infer_list_sound_helper` -- list soundness over `infer_list` (feeds `infer_type_sound`)
66. `infer_type_sound_inner` -- soundness via `expr_ind_strong` (public `infer_type_sound` instantiates it)
67. `has_type_list_det` -- list determinism over `Forall2 (has_type env)` (feeds `has_type_det`)
68. `has_type_det_inner` -- determinism via `expr_ind_strong` (public `has_type_det` instantiates it)
69. `infer_list_complete_helper` -- list completeness over `Forall2` (feeds `infer_type_complete`)
70. `infer_type_complete_inner` -- completeness via `expr_ind_strong` (public `infer_type_complete` instantiates it)
71. `pat_expr_ind_strong` -- custom strong induction principle for `pat_expr` (Forall IH over branch bodies); `Defined`
72. `check_branches_sound` -- branch-list soundness over `check_branches` (feeds `infer_pat_type_sound`)
73. `constr_expr_ind_strong` -- custom strong induction principle for `constr_expr` (Forall IH over provided fields + Some/None payload arm); `Defined`
74. `check_fields_sound` -- field-list soundness over `check_fields` (feeds `infer_constr_type_sound`)

## Next: T3-S9
