# TypeSafety — Status

**Branch**: formal/type-safety-phase1b
**Phase**: T2-UNIFY (done 2026-06-14) → T3-SEMANTIC (next)
**Toolchain**: Rocq 9.1.1 / OCaml 5.4.0

## Scoreboard

| Metric | Value |
|---|---|
| Theorems proven | 30 |
| Admits | 0 |
| Axioms | 0 |
| Definitions | infer_type, lookup_env, has_type, pre_type, follow, follow_pvar, occurs_in, unify_fun, apply_subst, + more |
| Build | green (CoqMakefile, exit 0; dune build/test, exit 0) |
| T1-CMBT harness | green — differential QCheck 2000/2000 (0 errors, 0 fails) + 20/20 smoke |
| T2-UNIFY harness | green — differential QCheck 1000/1000 (0 errors, 0 fails) + 15/15 smoke |
| Findings | F-TS-01 (ELet scope leak) — found by T1-CMBT, RESOLVED |

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

## Proven (tick 1-2, TypeSafetySpec.v — T1)

(see proof-ledger.json for full list of 9 T1 theorems)

## Proven (tick 3, UnifySpec.v — T2-UNIFY, all Qed, 0 admits)

21. `pre_type_ind_strong` — custom induction for pre_type (Forall IH for PTuple)
22. `follow_ground_prim/reg/tuple` — ground types are fixed by follow (3 lemmas)
23. `follow_pvar_none` — unbound PVar follows to itself
24. `prim_type_beq_true/false` — prim_type_beq reflects =/≠ (2 lemmas)
25. `reg_type_beq_true/false` — reg_type_beq reflects =/≠ (2 lemmas)
26. `prim_type_beq_refl` — prim_type_beq p p = true
27. `reg_type_beq_refl` — reg_type_beq r r = true
28. `subst_lookup_head` — head binding lookup
29. `occurs_ground_false` — PPrim/PReg contain no PVar (2 lemmas)
30. `unify_fun_prim_prim/reg_reg` — unfolding lemmas (2 lemmas)
31. `unify_fun_var_prim/var_reg` — conditional unfolding for PVar/ground (2 lemmas)
32. `unify_zero_none` — zero fuel always returns None
33. `unify_prim_sound/complete` — PPrim soundness and completeness
34. `unify_reg_sound/complete` — PReg soundness and completeness
35. `unify_var_binds_prim/reg` — free PVar gets bound to ground type
36. `occurs_check_blocks_prim/reg` — occurs_in ground always false

## Next: T3-SEMANTIC
