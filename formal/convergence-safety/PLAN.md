# ConvergenceSafety — Work Plan

**Last updated**: 2026-06-13 (tick 8 — T3-S4 done; currentTask = T3-S5)
**Apparatus version**: 1.2.1
**Phase**: T3-SEMANTIC (full ladder approved; T3-S5 is current task)

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1A-CONF | Extend QCheck conformance tests — 3 dedicated ESuperstep false/true QCheck properties | T1 | **done** (17/17 green, PR #182 merged 2026-06-12T06:53:15Z) | — |
| T1-LATEX | Reconcile ConvergenceSafetySpec.tex — add superstep_outer_diverged_error; verify all theorems listed | T1 | **done** (v0.4, 20 theorems, 2026-06-12) | — |
| T1-LEDGER | Sync proof-ledger.json — 20 theorems, summary.total=20 | T1 | **done** (20 entries, summary.total=20) | — |
| T2-F02 | Environment-threaded is_varying in Rocq spec for F-02 | T2 | **done** | — |
| T2-WARP | WarpConvergence error class: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags | T2 | **done** | — |
| T2-RETURN | TEReturn early-return barrier-skip: EReturn constructor, model barrier-skip, safety theorem | T2 | **done** (return_barrier_skip_safe, 20 theorems, 17 conformance, 7 extraction) | — |
| T3-GATE | HUMAN DECISION — approve T3-SEMANTIC scope per the breakdown below (whole ladder, or stop at SEMANTIC-CORE) | T3 | **done** (full ladder approved 2026-06-13) | — |
| T3-S1 | Semantic domain + fuel-indexed big-step evaluator with barrier traces | T3 | **done** (ConvergenceSemantics.v: eval + eval_fuel_monotone + eval_app_seq_compose; 22 theorems, 0 admits, 0 axioms, coqchk passes; 2026-06-13) | — |
| T3-S2 | Uniformity soundness of `is_varying_in_env` (semantic grounding of EVary) | T3 | **done** (ConvergenceSemantics.v: env_agrees + not_varying_uniform + closed_uniform; 27 theorems, 0 admits, 0 axioms; 2026-06-13) | T3-S1 |
| T3-S3 | Trace silence of barrier-free expressions | T3 | **done** (ConvergenceSemantics.v: no_barrier_event + superstep_free + barrier_free_no_barriers + diverged_clean_no_barriers; 33 theorems, 0 admits, 0 axioms; commit af9f6b81354e7c6d7c5779c273315bbd6a295eff; 2026-06-13) | T3-S1 |
| T3-S4 | Core semantic soundness of `check_env` (trace-uniformity theorem) | T3 | **done** (ConvergenceSemantics.v: core_frag + eval_check_uniform + check_env_sound_core; 36 proven, 6 defs, 0 admits, 0 axioms; 2026-06-13) | T3-S2, T3-S3 |
| T3-S5 | EReturn residual-divergence verdict (formal counterexample or proof; expected F-04) | T3 | open | T3-S4 |
| T3-S6 | ESuperstep semantic grounding (implicit-barrier event; semantic F-01) | T3 | open | T3-S4 |
| T3-S7 | Warp-collective semantic soundness (`check_warp` vs warp-granular traces) | T3 | open | T3-S4 |
| T3-S8 | Extraction + differential conformance for the semantics (CMBT closure) | T3 | open | T3-S5 |
| DOCS-SYNC | STATUS.md / ASSUMPTIONS.md / proof-ledger.json / ConvergenceSafetySpec.tex drift check | hygiene | **clean** (verified tick 1) | — |

---

## Current task

**T3-S5 — current task.**

T3-S4 done (2026-06-13): ConvergenceSemantics.v extended with core_frag (Fixpoint,
excludes ESuperstep/EReturn), erase_warp (Definition, projects trace to EvBarrier only),
barrier_safe (Definition, erase_warp equality for all env-agreeing thread pairs),
eval_check_uniform (Lemma, combined Part A barrier-trace + Part B outcome uniformity by
fuel induction), check_env_sound_core (Theorem, one-liner via proj1 eval_check_uniform);
36 proven, 6 defs, 0 admits, 0 axioms.
Design note: barrier_safe uses erase_warp equality (not full trace equality) so EWarpPoint
warp-collective events are tolerated; the simultaneous Part A/B induction is the "new
technique" cited in the T3-S4 plan as the reason for L effort.

Current task: T3-S5 — EReturn residual-divergence verdict (formal counterexample or proof; expected F-04).
Blocked-by T3-S4 is resolved.

---

## T3-SEMANTIC — concrete breakdown

### Global design decisions (apply to all subtasks)

1. **New file, frozen T2 artifact.** All semantic work lives in a new
   `theories/ConvergenceSemantics.v` that `Require Import`s `ConvergenceSpec`.
   `ConvergenceSpec.v` stays frozen (its 20 theorems are LOCKED); the only permitted
   change there is exporting definitions if visibility requires it.
2. **Big-step with fuel, not small-step.** The evaluator is a total `Fixpoint` on a
   fuel argument returning `option`. Rationale: no coinduction, no step-indexed
   logical relations, executable (extractable for differential CMBT), and `EWhile`
   nontermination is absorbed by `None`. Small-step + traces would force trace
   induction/coinduction (L+ everywhere) for no gain at this abstraction level.
3. **`EVary` as a Section parameter, not an axiom.** `Variable vary_val : tid -> value`
   inside a Section. All theorems quantify over it on closing the Section. Preserves
   the 0-axiom invariant (`coqchk` must keep passing with 0 axioms).
4. **Safety = barrier-trace uniformity, not operational deadlock.** A program is
   `barrier_safe` iff all threads produce the *same sequence of synchronization
   events*. Under the SIMT lockstep assumption this is equivalent to "no barrier is
   reached by a strict subset of threads", i.e. no barrier deadlock. Modeling the
   lockstep scheduler and proving deadlock-freedom operationally is deliberately OUT
   OF SCOPE (record in ASSUMPTIONS.md as the T3 trust-boundary statement). This is
   the single biggest cost-saving decision in the plan.
5. **Outcome type from day one.** The evaluator returns
   `outcome := ONorm value | ORet value` so that EReturn early-exit (needed by T3-S5)
   never forces a refactor of T3-S1..S4 proofs. T3-S4 simply restricts its statement
   to a return-free fragment.
6. **Pure-language quirk acknowledged.** The abstract `expr` has no mutation, so
   `EWhile` over a uniform condition runs 0 or ∞ iterations. This is faithful to the
   abstraction level of `ConvergenceSpec.v` (the OCaml checker is equally
   state-insensitive) — document, don't fix.

---

### T3-S1 — Semantic domain + fuel-indexed big-step evaluator with barrier traces

- **What to add** (`theories/ConvergenceSemantics.v`):
  - `Definition tid := nat.` `Definition value := nat.` (booleans via `0 = false`;
    minimal — no value sum type until something forces it)
  - `Definition venv := list (nat * value).` with `venv_lookup`, `venv_extend`
    (mirror `Env`/`env_lookup`/`env_extend` shape from `ConvergenceSpec.v`)
  - `Inductive event : Type := EvBarrier | EvWarp.` `Definition trace := list event.`
  - `Inductive outcome : Type := ONorm : value -> outcome | ORet : value -> outcome.`
  - Section with `Variable vary_val : tid -> value.`
  - `Fixpoint eval (fuel : nat) (t : tid) (ρ : venv) (e : expr) : option (outcome * trace)`
    — `EVary ↦ vary_val t`; `EBarrier ↦ (ONorm 0, [EvBarrier])`;
    `EWarpPoint ↦ (ONorm 0, [EvWarp])`; `EIf`/`EWhile`/`EFor` branch on the evaluated
    condition; `ELet` extends `ρ`; `ESeq`/`EApp` thread traces by `++` and
    short-circuit on `ORet`; `EReturn e ↦ ORet`-wrap; `ESuperstep dv body cont ↦`
    body trace ++ `[EvBarrier]` (implicit end-of-step barrier, dv-independent at
    runtime) ++ cont trace, with `ORet` propagation; `EWhile` consumes fuel.
  - Theorems: `eval_fuel_monotone : eval n t ρ e = Some r -> eval (S n) t ρ e = Some r`
    (the standard sanity lemma; everything later relies on it) and
    `eval_app_seq_compose` (trace homomorphism over `ESeq`, semantic mirror of
    `check_seq_hom`).
- **Key design decision**: decisions 2/3/4/5 above land here. Also: the implicit
  superstep barrier event is emitted for **both** `dv = true/false` (the runtime
  barrier exists regardless of the analysis flag — confirmed in
  `Sarek_convergence.ml` `expr_uses_barriers`, which returns `true` for every
  `TESuperstep`); the *checker* difference is only in what it flags. Verify against
  `Sarek_lower_ir.ml` during the subtask and record in ASSUMPTIONS.md.
- **Effort**: M
- **Blockers**: T3-GATE

### T3-S2 — Uniformity soundness of `is_varying_in_env`

- **What to add**:
  - `Definition env_agrees (env : Env) (ρ1 ρ2 : venv) : Prop :=`
    every variable mapped `false` in `env` has the same value in `ρ1` and `ρ2`
    (and is bound in both).
  - `Theorem not_varying_uniform : forall fuel env e t1 t2 ρ1 ρ2,`
    `env_agrees env ρ1 ρ2 -> is_varying_in_env env e = false ->`
    `eval fuel t1 ρ1 e = eval fuel t2 ρ2 e.`
  - Corollary for the binding-blind `is_varying` on closed (EVar-free) expressions.
- **Why it matters**: this is the *meaning* of `EVary`: it grounds the entire
  varying/uniform abstraction that T1/T2 took as primitive, and is the semantic
  counterpart of the F-02 env-threaded model. After this theorem, "varying" is no
  longer an unexplained tag.
- **Key design decision**: agreement is stated on env-flagged-uniform variables only
  (thread-private values may differ freely) — this is what makes the theorem usable
  inside T3-S4's `ELet` case.
- **Effort**: M (mechanical induction on fuel × expr; large case count, no new
  technique)
- **Blockers**: T3-S1

### T3-S3 — Trace silence of barrier-free expressions

- **What to add**:
  - `Theorem barrier_free_silent : forall fuel t ρ e o tr,`
    `barrier_free e = true -> eval fuel t ρ e = Some (o, tr) -> tr = [].`
  - `Corollary diverged_clean_silent : check Diverged e = [] -> ...tr = [].`
    (composes with the existing `diverged_clean_iff_barrier_free` — the T1 flagship
    theorem acquires semantic content here)
- **Key design decision**: none — pure structural induction. Note `barrier_free`
  already returns `false` for `ESuperstep false _ _`, and with the S1 decision to
  emit the implicit event for `dv = true` too, the `ESuperstep true` case makes
  `barrier_free` and runtime silence **disagree**. Resolution: state the theorem over
  superstep-free expressions (`superstep_free e = true` side condition) and leave the
  superstep case to T3-S6, OR weaken `tr = []` to "no `EvBarrier` from explicit
  `EBarrier` nodes". Decide in-task; the side-condition route is cheaper and honest.
- **Effort**: S
- **Blockers**: T3-S1

### T3-S4 — Core semantic soundness of `check_env`  ⚑ L, new proof technique

- **What to add**:
  - `Fixpoint core_frag (e : expr) : bool` — true iff `e` contains no `EReturn` and
    no `ESuperstep` (the fragment soundness is first proven on).
  - `Definition barrier_safe (env : Env) (e : expr) : Prop := forall fuel t1 t2 ρ1 ρ2 o1 o2 tr1 tr2,`
    `env_agrees env ρ1 ρ2 -> eval fuel t1 ρ1 e = Some (o1, tr1) -> eval fuel t2 ρ2 e = Some (o2, tr2) -> tr1 = tr2.`
  - **Main theorem**:
    `Theorem check_env_sound_core : forall env e, core_frag e = true ->`
    `check_env Converged env e = [] -> barrier_safe env e.`
  - Required inner lemma (the mode invariant):
    `check_env Diverged env e = [] -> (silent: every thread's trace is [])` — follows
    from T3-S3 + a `check_env`-to-`barrier_free` bridge lemma
    (`check_env_diverged_clean_barrier_free`, the env-threaded analogue of
    `diverged_clean_iff_barrier_free`; must be proven here, it does not yet exist).
- **Proof skeleton**: induction on expr (via `expr_list_rect`) under a fuel
  quantifier. `EIf` non-varying cond: T3-S2 gives both threads the same branch →
  IH composes traces. `EIf` varying cond: checker forces branches Diverged-clean →
  bridge lemma → T3-S3 → both branches silent → traces agree regardless of branch
  taken. Same shape for `EWhile`/`EFor`; `EWhile` additionally needs an inner
  induction on fuel to show iteration counts agree when the condition is uniform —
  **this is the genuinely new technique (simultaneous fuel + expr induction with a
  mode-indexed invariant)** and the reason this task is L.
- **Key design decision**: soundness is stated for `check_env` (the post-F-02-fix
  model), not the binding-blind `check` — the binding-blind checker is *known*
  semantically unsound (F-02), so only a fragment-restricted corollary for closed
  expressions is worth stating for `check`.
- **Effort**: L
- **Blockers**: T3-S2, T3-S3

> **── SEMANTIC-CORE milestone boundary (see below) ──**

### T3-S5 — EReturn residual-divergence verdict (expected new finding F-04)

- **What to add**:
  - `Definition hazard : expr := ESeq [EIf EVary (EReturn ELit) ELit; EBarrier].`
  - Theorem 1 (checker is blind):
    `check_env Converged env hazard = []` — by computation.
  - Theorem 2 (semantic divergence): exhibit `vary_val`, `t1`, `t2` with
    `vary_val t1 <> vary_val t2` such that thread `t1` (takes the return) yields
    trace `[]` and `t2` yields `[EvBarrier]` — i.e.
    `~ barrier_safe env hazard`.
  - Together: a machine-checked **counterexample to whole-language soundness** of the
    transparent-EReturn checker design, resolving the open audit item in
    ASSUMPTIONS.md ("residual divergence at a later barrier") with a formal verdict.
- **Expected outcome**: new finding **F-04** against `Sarek_convergence.ml`
  (`TEReturn` transparency is a false negative at kernel granularity: a thread that
  early-returns inside thread-varying control flow skips every later barrier). File
  in `findings/DIVERGENCE_FINDINGS.md`; OCaml fix (e.g. mark mode Diverged after a
  conditional-return join, or reject conditional returns before barriers) is a
  host-side follow-up, NOT part of this subtask.
- **Key design decision**: produce a *counterexample theorem*, not an extended
  soundness theorem. Sound checking of conditional returns needs path-sensitive
  may-return analysis — out of scope; the counterexample is what unblocks the audit
  item and motivates the host fix.
- **Effort**: M (the evaluator already short-circuits `ORet` from T3-S1; the proofs
  are computations on a concrete term)
- **Blockers**: T3-S4 (uses `barrier_safe`)

### T3-S6 — ESuperstep semantic grounding (semantic F-01)

- **What to add**:
  - Extend the fragment: `core_frag_ss` admits `ESuperstep` (still no `EReturn`).
  - `Theorem check_env_sound_superstep` — T3-S4 statement over the enlarged fragment.
    New case: superstep entered under uniform reachability emits `[EvBarrier]` on
    *all* threads (traces still agree); entry under thread-varying reachability is
    exactly what the checker's `Diverged, false => [BarrierError]` entry error
    excludes.
  - Semantic F-01 corollary: an `ESuperstep false` reachable by only a subset of
    threads (encode as `EIf EVary (ESuperstep false b c) ELit`) is **not**
    `barrier_safe` — and `check_env` flags it (the static F-01 theorem
    `superstep_outer_diverged_error` finally gets its runtime meaning).
  - Resolve the T3-S3 side condition (`superstep_free`) by proving the strengthened
    silence lemma for the enlarged fragment.
- **Key design decision**: whether `ESuperstep true` (divergent flag) also emits the
  implicit event was fixed in T3-S1 (it does); here decide how the *checker's*
  `dv = true` permissiveness is reconciled — expected answer: `dv = true` supersteps
  are excluded from the soundness fragment (the flag is an explicit programmer
  opt-out, document as trust boundary in ASSUMPTIONS.md).
- **Effort**: M
- **Blockers**: T3-S4

### T3-S7 — Warp-collective semantic soundness

- **What to add**:
  - Section `Variable warp_of : tid -> nat.` (warp partition; warp size never
    appears — full parameterization, closing the "warp size parameterization
    remains T3" item in ASSUMPTIONS.md)
  - `Definition warp_safe (env : Env) (e : expr) : Prop :=` trace agreement
    restricted to thread pairs with `warp_of t1 = warp_of t2`.
  - `Theorem check_warp_sound_core : core_frag e = true ->`
    `check_warp_env Converged env e = [] -> warp_safe env e`
    (needs an env-threaded `check_warp_env` first — small definition, mirrors
    `check_env` + `EWarpPoint` case; add it here, plus its Diverged-clean bridge).
- **Key design decision**: attempt to make the T3-S4 induction *parametric* over
  (event class, agreement domain, checker) so this is instantiation rather than
  re-proof. If T3-S4 was not written parametrically, budget rises to L — flag at
  task start and decide whether to refactor S4 or duplicate the induction.
- **Effort**: M (risk: L if no reuse)
- **Blockers**: T3-S4

### T3-S8 — Extraction + differential conformance for the semantics (CMBT closure)

- **What to add** (not in ConvergenceSpec.v — test/ + extraction config):
  - Extract `eval` (with `vary_val` instantiated to e.g. `fun t -> t`) into the
    `ConvergenceModel` extraction module.
  - QCheck properties in `test/test_convergence_conformance.ml` (or a new
    `test_convergence_semantics.ml`): (1) `eval_fuel_monotone` random check;
    (2) `barrier_free_silent` random check; (3) the headline differential —
    random core-fragment `expr` with `check_env Converged = []` ⇒ traces of two
    random tids/envs agree; (4) F-04 hazard regression (counterexample stays a
    counterexample).
  - DOCS-SYNC: STATUS.md theorem table, proof-ledger.json, ConvergenceSafetySpec.tex
    (new §Semantics), ASSUMPTIONS.md trust-boundary updates from decisions 4, S1,
    S6.
- **Key design decision**: differential testing runs the *extracted Rocq evaluator*
  against the *Rocq checker verdict* — it validates the extraction TCB, not the
  OCaml host (the host has no evaluator to test against; the semantics-to-host link
  remains the documented model-faithfulness assumption).
- **Effort**: M
- **Blockers**: T3-S5 (covers everything up to and including F-04; if S6/S7 land
  first, fold their properties in)

---

### Natural stopping point — SEMANTIC-CORE milestone

**After T3-S5** the project is *semantically grounded*:

- `check_env` is proven sound against a real execution semantics on the core
  fragment (T3-S4) — "checker says clean" now *means* "all threads cross the same
  barriers".
- `EVary`/`is_varying` and `barrier_free` are no longer unexplained abstractions
  (T3-S2, T3-S3).
- The only known open soundness question (EReturn residual divergence) is formally
  adjudicated (T3-S5 / F-04).

Everything after (S6 superstep, S7 warp, S8 CMBT closure) widens coverage but does
not change the epistemic status of the project. Hardware connection (lockstep
scheduler, memory model, real deadlock-freedom) remains permanently out of scope per
global decision 4 and is recorded as the T3 trust boundary.

---

## Blocker rules (standing)

- T3-GATE is a human gate — workflow pauses and surfaces the scope question
  (full ladder vs SEMANTIC-CORE vs defer).
- T3-S1 is blocked by T3-GATE; each T3-Sn is blocked per the table above.
- T3-S4 is the long pole and the only L task — do not start T3-S5/S6/S7 speculatively
  before it lands (they all consume `barrier_safe` and the bridge lemma).
- If T3-S5 confirms F-04: file the finding, surface to operator (host-side OCaml fix
  is a separate decision), do NOT silently extend scope.
- `ConvergenceSpec.v` is frozen during T3 — all additions go to
  `theories/ConvergenceSemantics.v` (and a possible `check_warp_env` exception
  decided in T3-S7).
- 0-axiom invariant holds throughout: Section Variables only; `coqchk` after every
  subtask.

---

## Workflow notes

- Tick 8 (2026-06-13): T3-S4 confirmed DONE. ConvergenceSemantics.v extended with
  core_frag, erase_warp, barrier_safe, eval_check_uniform (combined Part A/B fuel induction),
  check_env_sound_core; 36 proven, 6 defs, 0 admits, 0 axioms. Key design: barrier_safe uses
  erase_warp equality; check_env_sound_core proved as one-liner via proj1 (eval_check_uniform).
  proof-ledger.json updated (total=42, proven=36, definitions=6). STATUS.md updated.
  PLAN.md promoted to T3-S5. currentTask = T3-S5 (unblocked — T3-S4 done).
- Tick 7 (2026-06-13): T3-S3 confirmed DONE. ConvergenceSemantics.v extended with
  no_barrier_event, superstep_free, no_barrier_app, for_loop_fixed_no_barrier,
  eval_seq_no_barrier, eval_args_no_barrier, barrier_free_no_barriers,
  diverged_clean_no_barriers; 33 theorems, 0 admits, 0 axioms. Key deviation: tr = []
  weakened to no_barrier_event tr = true (EWarpPoint emits [EvWarp] but is barrier_free).
  proof-ledger.json updated (total=38, proven=33, definitions=5). STATUS.md updated.
  PLAN.md promoted to T3-S4. ConvergenceSafetySpec.tex updated with T3-S3 section.
  currentTask = T3-S4 (unblocked — T3-S2 and T3-S3 both done).
- Tick 5 (2026-06-13): Session re-read. All tasks through T3-S1 confirmed DONE (STATUS.md: 22
  theorems, 0 admits, 0 axioms, coqchk passes; conformance 17/17 green, extraction 7/7 green,
  live 10/10 green). PR #182 confirmed merged (gh returns []). DOCS-SYNC clean — no drift
  detected. currentTask = T3-S2 (unblocked — T3-S1 done, no hard blockers).
- Tick 4 (2026-06-13): T3-S1 confirmed DONE (commit fbfb3656; ConvergenceSemantics.v:
  semantic domain + fuel-indexed big-step eval + eval_fuel_monotone + eval_app_seq_compose;
  22 theorems, 0 admits, 0 axioms, coqchk passes; STATUS.md updated).
  DOCS-SYNC: PLAN.md drift fixed (T3-S1 was listed as "open"; corrected to done).
  currentTask = T3-S2 (unblocked — T3-S1 done).
- Tick 3 (2026-06-13): T3-GATE resolved — full ladder T3-S1..S8 approved by human.
  currentTask = T3-S1 (unblocked). Workflow ready to fire.
- Tick 2 (2026-06-12): T3-SEMANTIC replanned from a one-line stub into 8 strictly
  ordered subtasks (T3-S1..S8) after a ground-truth re-read of
  `Sarek_convergence.ml` and `ConvergenceSpec.v`. Key decisions: fuel-indexed
  big-step evaluator (no coinduction); safety = barrier-trace uniformity (deadlock
  modeling out of scope, trust-boundary documented); `vary_val` as Section parameter
  (0-axiom invariant preserved); outcome type (`ONorm`/`ORet`) in the evaluator from
  day one so EReturn never forces a refactor; T3-S5 expected to yield finding F-04
  (EReturn transparency is a kernel-granularity false negative — formal
  counterexample, resolves the open ASSUMPTIONS.md audit item). SEMANTIC-CORE
  milestone defined at end of T3-S5. T3-GATE remains open with a now-concrete scope
  question.
- Tick 1 (2026-06-12): T1A-CONF executed and confirmed DONE. PR #182 merged
  (confirmed via `gh pr view 182 --repo mathiasbourgoin/Sarek`; merged 2026-06-12T06:53:15Z).
  Added 3 ESuperstep QCheck properties to test_convergence_conformance.ml:
  `superstep_outer_diverged_error` (F-01 direct), `superstep_no_entry_error_converged`
  (safe-path complement), `superstep_body_errors_propagate` (monotonicity).
  17/17 conformance properties GREEN. STATUS.md updated. DOCS-SYNC clean.
  currentTask promoted to T3-GATE (human gate — workflow stops).
- Tick 0 (2026-06-12, fresh session): T2-RETURN confirmed DONE (STATUS.md: 20 theorems,
  return_barrier_skip_safe, 0 admits, 0 axioms, coqchk passes; 14 conformance props, 7 extraction,
  all green). T1-LATEX confirmed DONE (ConvergenceSafetySpec.tex v0.4, 20 theorem environments,
  LOCKED 2026-06-12). T1-LEDGER confirmed DONE (proof-ledger.json: 20 entries, summary.total=20).
  DOCS-SYNC clean — no drift detected across ledger/LaTeX/STATUS.
  PR #182 STILL OPEN — T1A-CONF remained hard-blocked.
  currentTask = T1A-CONF (awaiting PR #182 merge to execute).
- Prior tick 1 (2026-06-12): T2-WARP confirmed DONE. T1-LEDGER confirmed DONE (19 entries).
  T1-LATEX re-opened (drift: 17 LaTeX envs vs 19 in ledger). currentTask promoted to T2-RETURN.
- Prior tick 0 (2026-06-12): T1-LATEX confirmed DONE (v0.2 LOCKED, 16 theorems). T1-LEDGER DONE.
  T2-F02 DONE. PR #182 STILL OPEN. currentTask promoted to T2-WARP.
- Prior tick 0 (2026-06-11): T2-F02 discovered DONE; T1-LATEX promoted as currentTask; T1-LEDGER queued.
