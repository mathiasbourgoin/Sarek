# Phase 1a Plan — Extend ConvergenceSpec.v for Confirmed Gaps

**Scope**: Rocq spec changes for F-01 (ESuperstep) + associated OCaml regression tests.
F-02 is a T2 dataflow task; Phase 1a adds a formal assumption documenting the known
false-negative rather than fixing it.

**Target file**: `/home/mathias/dev/SPOC/formal/convergence-safety/theories/ConvergenceSpec.v`

---

## Task P1A-1 — Add ESuperstep constructor to `expr`

**What to change in ConvergenceSpec.v**:

At line 29 (after `ELet`), add:

```coq
| ESuperstep : bool -> expr -> expr -> expr  (* divergent_flag, body, cont *)
```

The `bool` mirrors `TESuperstep`'s `divergent` flag in the real AST.
`body` corresponds to `step_body`, `cont` to the continuation.

**Rationale**: `TESuperstep` unconditionally emits `SBarrier` after `body` at lowering
time. The checker must therefore treat the step boundary as an implicit `EBarrier` for
safety analysis purposes, contingent on outer mode.

---

## Task P1A-2 — Extend `is_varying` for ESuperstep

Add case to `is_varying` fixpoint (after `ELet` case, ConvergenceSpec.v line 47):

```coq
| ESuperstep _ body cont => is_varying body || is_varying cont
```

**Rationale**: structural — a superstep expression is varying if any sub-expression
varies. This matches the pattern of all other compound constructors.

---

## Task P1A-3 — Extend `barrier_free` for ESuperstep

Add case to `barrier_free` fixpoint (after `ELet` case, ConvergenceSpec.v line 63):

```coq
| ESuperstep _ body cont => barrier_free body && barrier_free cont
```

Note: `ESuperstep` with `divergent=false` introduces an *implicit* barrier at the
step boundary that is not represented as a syntactic `EBarrier` child. Phase 1a
adds a separate `check` rule (P1A-5) to emit `BarrierError` at the entry point when
`m = Diverged`; `barrier_free` only covers syntactic barriers in subexpressions.
A comment in the spec should document this distinction.

---

## Task P1A-4 — Extend `has_diverging_cf` for ESuperstep

Add case to `has_diverging_cf` fixpoint (after `ELet` case, ConvergenceSpec.v line 77):

```coq
| ESuperstep _ body cont => has_diverging_cf body || has_diverging_cf cont
```

**Rationale**: mirrors `contains_diverging_control_flow` lines 307–309 of
`Sarek_convergence.ml`, which recurses into nested supersteps. The abstract predicate
must follow the same structural descent.

---

## Task P1A-5 — Add ESuperstep case to `check`

Add case to `check` fixpoint (after `ELet` case, ConvergenceSpec.v line 101):

```coq
| ESuperstep divergent body cont =>
    (* The implicit barrier fires at the end of every superstep body.
       Non-divergent superstep: entry under Diverged outer mode is an error
       (the implicit barrier is reached by only some threads).
       Divergent superstep: no implicit-barrier error at entry; the divergent
       flag signals that the checker already accepted inter-thread variation inside.
       In both cases the continuation is always entered Converged (re-sync). *)
    let entry_errors :=
      match m, divergent with
      | Diverged, false => [BarrierError]
      | _,        _     => []
      end
    in
    entry_errors ++ check m body ++ check Converged cont
```

**Key design decisions**:
- For `ESuperstep false body cont` with `m = Diverged`: emit `BarrierError` immediately
  (the implicit barrier is reached under diverged flow), then still check `body` with `m`
  to catch any additional barriers inside the body, then check `cont` with `Converged`.
- For `ESuperstep true body cont`: no extra error at entry (divergent superstep is
  expected to have per-thread paths); check `body` with `m`; check `cont` with
  `Converged`.
- The `cont` always uses `Converged` — this captures the re-synchronisation invariant
  that the implicit end-of-superstep barrier reconverges all threads.

---

## Task P1A-6 — Add IH_superstep to ExprListInd

In the `ExprListInd` section (ConvergenceSpec.v lines 129–176), add a new hypothesis:

```coq
Hypothesis IH_superstep : forall (dv : bool) body cont,
  P body -> P cont -> P (ESuperstep dv body cont).
```

And a new match arm in `expr_list_rect`:

```coq
| ESuperstep dv body cont =>
    IH_superstep dv body cont
      (expr_list_rect body) (expr_list_rect cont)
```

---

## Task P1A-7 — Re-prove diverged_clean_iff_barrier_free

Add the `ESuperstep` case to the proof (ConvergenceSpec.v lines 260–299):

```coq
(* ESuperstep dv body cont *)
- intros dv body cont IHbody IHcont. simpl.
  destruct dv, IHbody, IHcont;
  rewrite app_nil_iff, andb_true_iff; tauto.
```

The case split on `dv` handles the `entry_errors` match:
- `dv = false`, `m = Diverged`: `entry_errors = [BarrierError]`, so `check Diverged
  (ESuperstep false body cont)` can only be `[]` if `BarrierError :: _` is empty →
  impossible. This means `check Diverged (ESuperstep false body cont) = []` is false,
  consistent with `barrier_free (ESuperstep false body cont)` being vacuously about
  syntactic barriers in children only. A careful case analysis is needed; the key
  insight is that `barrier_free` does not track the implicit barrier, so the theorem
  requires a note or auxiliary lemma distinguishing syntactic from implicit barriers.

**Note**: After adding the implicit-barrier emitting rule, `diverged_clean_iff_barrier_free`
needs a clarifying comment: the equivalence `check Diverged e = [] <-> barrier_free e = true`
holds for the extended spec only when `e` contains no `ESuperstep false` nodes at depth
reachable under `Diverged` mode — because `ESuperstep false` emits `BarrierError` from
the implicit barrier even when `barrier_free` (which only checks syntactic `EBarrier`
children) returns `true`. Either (a) extend `barrier_free` to treat `ESuperstep false` as
containing an implicit barrier, or (b) restrict the theorem's domain with a side condition
`no_implicit_barriers e`. Option (a) is cleaner.

**Revised approach for `barrier_free`** (P1A-3 amendment):

```coq
| ESuperstep divergent body cont =>
    (* A non-divergent superstep contains an implicit barrier at its boundary *)
    (if divergent then true else false) && barrier_free body && barrier_free cont
```

This makes `barrier_free (ESuperstep false _ _) = false` always, so
`diverged_clean_iff_barrier_free` holds without a side condition.

---

## Task P1A-8 — Re-prove mode_monotone for ESuperstep

Add the `ESuperstep` case to the proof (ConvergenceSpec.v lines 303–351):

```coq
(* ESuperstep dv body cont *)
- intros dv body cont IHbody IHcont. simpl.
  destruct dv.
  + (* true: entry_errors = [] in both modes; cont always Converged *)
    apply incl_app_mono; [apply incl_app_mono; [apply incl_refl | exact IHbody] |
                          apply incl_refl].
  + (* false: Converged→entry_errors=[]; Diverged→entry_errors=[BarrierError] *)
    simpl. apply incl_app_mono; [apply incl_nil_l |
           apply incl_app_mono; [exact IHbody | apply incl_refl]].
```

---

## Task P1A-9 — Re-prove cdcf_check_agreement for ESuperstep

Add the `ESuperstep` case to the proof (ConvergenceSpec.v lines 407–459):

```coq
(* ESuperstep dv body cont *)
- intros dv body cont IHbody IHcont H. simpl in *.
  apply orb_false_iff in H. destruct H as [Hbody Hcont].
  rewrite (IHbody Hbody), (IHcont Hcont).
  (* entry_errors: has_diverging_cf = false implies no diverged control flow reached
     the superstep entry — mode is Converged, so entry_errors = [] *)
  simpl. reflexivity.
```

The key step: if `has_diverging_cf (ESuperstep dv body cont) = false`, then neither
`body` nor `cont` contain diverging CF. By `cdcf_check_agreement` applied inductively,
`check Converged body = []` and `check Converged cont = []`. The `entry_errors` match
with `m = Converged` always produces `[]`. So the whole expression checks clean.

---

## Task P1A-10 — Add new theorem superstep_outer_diverged_error

```coq
Theorem superstep_outer_diverged_error : forall body cont,
  check Diverged (ESuperstep false body cont) <> [].
Proof.
  intros body cont. simpl. discriminate.
Qed.
```

This is the central safety property for F-01: entering a non-divergent superstep in
`Diverged` mode always produces at least one `BarrierError`.

---

## Task P1A-11 — Add admitted assumption for F-02 (do NOT fix in Phase 1a)

Add a comment block and admitted assumption to ConvergenceSpec.v after the `check`
definition:

```coq
(* KNOWN LIMITATION F-02: binding-blind is_varying.
 *
 * The abstract ELet case `is_varying (ELet v b) = is_varying v || is_varying b`
 * mirrors the real checker's behaviour (Sarek_convergence.ml line 86, 199-200):
 * TEVar(name,_) is tested against a static primitive table; let-bound user-defined
 * variables always return false.
 *
 * Consequence: `let x = thread_idx_x in if x > 0 then barrier()` is a false negative
 * in both the real checker and this abstract model.
 *
 * A sound fix requires an environment-threaded is_varying (T2 task).
 * The current theorems (diverged_clean_iff_barrier_free, mode_monotone,
 * cdcf_check_agreement, varying_if_flags_barriers) are internally consistent
 * for the abstract model as defined; they do not guarantee soundness for programs
 * using let-bound aliases to thread-varying values.
 *)
```

Also add a cross-reference comment at line 86 of `Sarek_convergence.ml`:
```ocaml
(* KNOWN LIMITATION F-02: binding-blind — see convergence-safety/findings/DIVERGENCE_FINDINGS.md *)
| TEVar (name, _) -> Sarek_core_primitives.is_thread_varying name
```

---

## OCaml regression tests (write when SPOC integration is available)

### Test R-01: F-01 — superstep under diverged outer mode

```ocaml
let () =
  (* let body = TEIntrinsicFun(barrier_ref, Some ConvergencePoint, []) in
     let superstep = TESuperstep("s", false, body, TEUnit) in
     let cond = TEIntrinsicConst thread_idx_x_ref in
     let prog = TEIf(TEBinop(Gt, cond, TEInt 16l), superstep, TEUnit) in
     assert (check_expr {mode=Converged} prog <> []) *)
  (* Expected: BarrierError emitted (currently: [] — confirmed false-negative) *)
  ()
```

Lock: once `check_expr` is fixed, this test must return `Error [_]`. Until fixed,
assert it returns `Ok ()` to lock the known false-negative.

### Test R-02: F-02 — let-aliased thread-varying variable as branch condition

```ocaml
let () =
  (* let prog = TELet("x", 0, TEIntrinsicConst thread_idx_x_ref,
                  TEIf(TEBinop(Gt, TEVar("x",0), TEInt 0l),
                       TEIntrinsicFun(barrier_ref, Some ConvergencePoint, []),
                       TEUnit)) in
     (* Known false-negative: assert currently returns Ok () *)
     assert (check_expr {mode=Converged} prog = []) *)
  (* TODO Phase 1a: flip to assert (check_expr ... <> []) once F-02 is fixed *)
  ()
```

### Test R-03: F-01 — superstep under varying for-loop bounds

```ocaml
let () =
  (* Non-divergent superstep inside a for-loop with thread-varying lo bound.
     Loop body is checked Diverged; superstep entry under Diverged must error. *)
  ()
```

---

## Estimated effort

| Task | Effort | Notes |
|---|---|---|
| P1A-1 through P1A-6 (constructor + fixpoints + IH) | ~1h | Mechanical; follow existing patterns |
| P1A-7 (diverged_clean_iff_barrier_free re-proof) | ~2h | Requires careful implicit-barrier/barrier_free reconciliation |
| P1A-8 (mode_monotone re-proof) | ~30m | Standard case split |
| P1A-9 (cdcf_check_agreement re-proof) | ~30m | Follows existing structure |
| P1A-10 (new theorem) | ~15m | Trivial once check case is right |
| P1A-11 (F-02 admitted comment) | ~15m | Commentary only |
| OCaml regression tests R-01 through R-03 | ~1h | Blocked on SPOC integration |
| `coqchk` verification pass | ~15m | Standard gate |
| **Total** | **~5.5h** | Excludes SPOC integration blocker |

---

## Dependencies and gates

- All Rocq tasks (P1A-1 through P1A-10) are unblocked and can proceed immediately.
- OCaml regression tests (R-01 through R-03) are blocked on SPOC test suite
  integration; they should be written as stubs that assert the current (buggy)
  behaviour so that regressions are detectable when the fix lands.
- After Phase 1a, run `/formal-check` before locking the extended spec.
- `coqchk` must pass with 0 axioms before closing Phase 1a.
