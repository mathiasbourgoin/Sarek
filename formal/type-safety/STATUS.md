# TypeSafety — Status

**Branch**: formal/type-safety-phase1c
**Phase**: T2-VEC (done 2026-06-14) -> T3-SEMANTIC (next)
**Toolchain**: Rocq 9.1.1 / OCaml 5.4.0

## Scoreboard

| Metric | Value |
|---|---|
| Theorems proven | 41 |
| Admits | 0 |
| Axioms | 0 |
| Definitions | infer_type, lookup_env, has_type, pre_type, follow, follow_pvar, occurs_in, unify_fun, apply_subst, infer_mem_type, sarek_type_eq_dec, has_mem_type, + more |
| Build | green (CoqMakefile, exit 0; dune build/test, exit 0) |
| T1-CMBT harness | green -- differential QCheck 2000/2000 (0 errors, 0 fails) + 20/20 smoke |
| T2-UNIFY harness | green -- differential QCheck 1000/1000 (0 errors, 0 fails) + 15/15 smoke |
| T2-VEC harness | green -- 10/10 smoke (EVecGet/EVecSet/EArrGet/EArrSet + error cases) |
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

## Next: T3-SEMANTIC
