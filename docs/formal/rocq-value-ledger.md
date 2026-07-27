# Rocq value ledger

**What this is.** A running record of what the Rocq formalisation under `formal/`
has actually caught, what it has missed, and what each of those was worth. One
row per event, with a counterfactual: *would this have been found otherwise, and
how?*

**Why it exists.** Three Rocq projects, 280 machine-checked theorems, and a CI job
that rebuilds and kernel-re-checks all of them are a real ongoing cost. Whether
that cost is repaid is an empirical question, and until this file existed nobody
was collecting the evidence either way — so the argument for and against the work
was conducted entirely on conviction. The findings below were not hidden; they
were recorded in four different places, in four different formats, and never
counted.

**This ledger is not an advocacy document.** A ledger that records only hits is
marketing. Misses are recorded with the same care as catches, and a verification
investment that produced no defect is recorded as producing no defect — see
[N-01](#n-01), which is the outcome of the most recent piece of work here.

---

## Admission rules

A row is a **catch** only if the formalisation is what surfaced it. Specifically,
one of:

- a proof would not close, and the reason was a real defect (`F-04`);
- a differential/conformance run against the extracted model produced a
  counterexample (`F-TS-01`);
- extending the model to cover a construct exposed that the implementation
  handled it wrongly (`F-01`, `F-03`).

A defect that was already known, already filed, or already commented in the source
as a deferred limitation is **not** a catch. The formalisation may have promoted
it to a tracked, regression-covered finding — that has value, and it is recorded
as a *promotion*, not a discovery (`F-02`).

Every row states a **counterfactual**. "It would have been hard to find otherwise"
is not a counterfactual; naming the specific mechanism that would or would not
have found it is. Where the honest answer is *code review would plausibly have
caught this*, that is what the row says.

Rows are appended, never rewritten. A row whose assessment later turns out to be
wrong gets a follow-up row.

---

## Baseline, measured 2026-07-27

Before this file existed the working assumption on record was **zero catches**.
That assumption was never measured; it was the absence of a count, not a count of
zero. Measured against the sources below, the baseline is:

| | count |
|---|---|
| implementation defects caught, fixed, regression-covered | **3** (F-01, F-04, F-TS-01) |
| specification gaps caught (model incomplete, no shipped defect) | **1** (F-03) |
| known limitations promoted to tracked findings | **1** (F-02) |
| defect classes in scope of a model and missed by it | **1** (M-01) |
| defect classes outside any model's scope | **1** (M-02) |
| verification work completed that found nothing | **1** (N-01) |

Sources consolidated: `formal/convergence-safety/findings/DIVERGENCE_FINDINGS.md`,
`formal/type-safety/findings/FINDINGS.md`,
`formal/type-safety/cmbt-record.json`,
`formal/codegen-ptx/findings/FINDINGS.md` (empty).

---

## Catches

### F-TS-01 — `let` scope leak in the production typer

| | |
|---|---|
| Date | 2026-06 (T1-CMBT) |
| Severity | MAJOR |
| Kind | implementation defect |
| Where | `sarek/ppx/Sarek_typer.ml` (`infer_let_binding`) |
| Found by | differential QCheck, extracted Rocq model vs the production typer |
| Counterexample | `let y = (let x = 0 in 0) in x` |
| Status | fixed, `20b79b36`; regression in `test_type_safety_conformance.ml` |

Inner `let` bindings inside a let-*value* leaked into the continuation body: the
typer accepted `x` as in scope where correct lexical scoping rejects it as
unbound. The body environment was built from the environment accumulated during
value inference instead of the pre-value one.

**Counterfactual — strong.** This is a defect of *over-acceptance*: the compiler
accepts a program it should reject. Hand-written tests almost never cover
"programs that must fail to typecheck" beyond the cases the author already has in
mind, and this one requires a `let` nested in a `let`-value with a use in the
continuation. Fuzzing alone would not have found it either, because a fuzzer needs
an oracle to tell it that acceptance was wrong — and the Rocq model *was* that
oracle. This is the case the CMBT apparatus is for, and it is the strongest single
entry in this ledger.

### F-04 — early return can skip a later barrier, undetected

| | |
|---|---|
| Date | 2026-06 (T3-S5) |
| Severity | soundness gap in the convergence checker |
| Kind | implementation defect (+ consequent spec extension) |
| Where | `sarek/ppx/Sarek_convergence.ml`, `TEReturn` handling |
| Found by | proof work — the soundness statement would not close |
| Status | fix landed + regression-covered; spec extended and re-proved (F-04b) |

The checker treats `EReturn` as a transparent wrapper, so it reasons about an
early return purely through the returned expression and never accounts for the
control-flow effect of the return itself: a thread that returns early under
varying flow skips every subsequent statement, including a later barrier.

**Counterfactual — strong.** Nothing about the code looks wrong; the transparency
is a deliberate, locally-reasonable choice, and the theorem
`return_barrier_skip_safe` states it explicitly. What exposed it was attempting to
prove kernel-granularity soundness and finding the obligation unprovable. Neither
review nor testing has a natural path to "this rule is sound at statement
granularity and unsound at kernel granularity"; that distinction only becomes
visible when a proof forces the granularity to be stated.

### F-01 — `ESuperstep` discards inherited diverged context

| | |
|---|---|
| Date | 2026-06 |
| Severity | missed divergence (false negative) |
| Kind | implementation defect, surfaced by a spec gap |
| Where | `sarek/ppx/Sarek_convergence.ml:231` |
| Found by | extending the abstract model to cover `ESuperstep` |
| Status | fixed + regression-covered (`test_f01_superstep_in_diverged_if`) |

The non-divergent `TESuperstep` branch unconditionally builds a fresh `Converged`
context for the step body, so an already-`Diverged` caller mode is silently
discarded and the implicit end-of-superstep barrier is never flagged.

**Counterfactual — moderate.** The defect is two adjacent lines, and the sibling
divergent path immediately above threads `ctx` correctly. A reviewer comparing the
two branches could see it. What the formalisation supplied was the *reason to
look*: adding the constructor to the model forced someone to state what the
context should be in each branch. Call this a catch, but not one that only a proof
could have made.

### F-03 — a whole error class was missing from the model

| | |
|---|---|
| Date | 2026-06 |
| Severity | specification incompleteness |
| Kind | spec gap — **no shipped defect** |
| Where | `theories/ConvergenceSpec.v` (model), mirroring `Sarek_convergence.ml:144` |
| Status | model extended (`EWarpPoint`, `WarpError`, `check_warp`), proven |

The real checker emits `Warp_collective_in_diverged_flow` as a distinct error
class; the abstract model had only `BarrierError`. Any theorem claiming `check`
exhausts all convergence errors was therefore false as stated.

**Counterfactual — n/a for shipped code; real for the guarantee.** The
implementation was correct throughout. What was wrong was the *claim*: a
completeness theorem about a model missing an error constructor is a completeness
theorem about nothing. Recorded because a value ledger that counts only bugs in
the product will systematically under-report the failure mode this work is most
exposed to — proving true things about the wrong model.

## Promotions

### F-02 — `is_thread_varying` is binding-blind

| | |
|---|---|
| Date | 2026-06 |
| Kind | **promotion, not discovery** |
| Where | `sarek/ppx/Sarek_convergence.ml:86`, `Sarek_core_primitives.ml:732` |
| Status | fixed + regression-covered |

`is_thread_varying` looks a name up in a table of statically-known intrinsics, so
any let-bound alias of a thread-varying value returns `false` and the varying-ness
is lost.

**Not counted as a catch.** The source already carried the comment *"Future:
varying_vars : StringSet.t for dataflow analysis"* at the exact site — the
limitation was known and deliberately deferred before any of this work started.
The formalisation's contribution was to turn a deferred TODO into a filed finding
with three regression tests and a fix. That is worth recording and it is not a
discovery, and conflating the two is how these ledgers stop being believed.

## Misses

### M-01 — the wrong-width family, in scope and missed

| | |
|---|---|
| Date | 2026-07 |
| Instances | 5 in a single day (helper return type, variant payload, `Char` stride, …) |
| Tracked as | audit 2026-07-24 items backlog-55, backlog-244, backlog-261 |

Element widths and aggregate layout are exactly what `formal/codegen-ptx`
(`PtxLayout.v`) and `formal/type-safety` (`TypeSafetySpec.v`) are about, and five
instances of a width defect shipped anyway.

**Why it was missed, specifically.** `PtxLayout.v` abstracts the scalar universe
to `lty = L32 | L64` — byte size and natural alignment, nothing else — and reasons
about *offsets given widths*. `TypeSafetySpec.v` models the typer, and stops at
the typer. The defective code is `elttype_of_typ`, the placeholder that maps an
OCaml type to a width, which sits in the lowering *between* the two models.

**And that gap is not an accident of scope — it is a design decision, recorded in
the theory's own header:**

> This theory deliberately does NOT import or extend `PtxTypes.elttype`; it
> defines its own small scalar universe `lty` carrying only what layout needs
> (byte size / natural alignment).
> — `formal/codegen-ptx/theories/PtxLayout.v`, lines 15–17

So `PtxLayout.v` does not merely fail to reach the width mapping. It **takes
widths as given, and therefore discards exactly the information the defective
code gets wrong.** The decoupling is good theory design — it is what keeps the
layout theorems independent of the type language — and it is also, precisely, the
hole. Both models are sound. Neither was ever going to see this.

**Lesson — and it is not "build more models".** That would be the platitude, and
it points the wrong way: the two models here are the deep, interesting ones, and
making them deeper would not have caught a single one of the five defects. What
was missing is a model of `elttype_of_typ` itself — a total function from a
finite type language to a byte width, with no theorems worth stating about it,
which an exhaustive table would settle in an afternoon.

**Cheap models of boring functions at module boundaries appear to be worth more
per line here than deep theorems about the ends.** The boring function is where
the defects were, and it is boring precisely because the interesting models
abstracted it away — which is the same reason nobody modelled it.

**That model now exists in executable form, on both halves of the seam** (backlog-141):

- `sarek/tests/unit/test_type_width_totality.ml` — source type → IR element
  type, the `elttype_of_typ` this entry names;
- `sarek/tests/codegen_golden/test_backend_type_width_totality.ml` — IR element
  type → device type string, for all six backends, against
  `Sarek_ir_layout.scalar_size`.

Each enumerates its finite type language by unfolding a **wildcard-free**
successor chain, so a new constructor is a compile error *in the test* rather
than a silently unswept case. Their contracts are not identical, and the
difference matters to anyone reading them as models:

- the **front half** is two-outcome — `elttype_of_typ` must either preserve the
  byte width or reject with a located error;
- the **backend half** is three-outcome — the emitted device type must occupy
  exactly `Sarek_ir_layout.scalar_size` bytes, or the mapper must refuse with a
  diagnostic, or the device type must be recorded as having **no memory form at
  all** and thereby exempted from the width check.

That third outcome is a real escape hatch and is treated as one. Its complete
set is pinned in `expected_no_memory_form` — six entries today, for two
different reasons: `TUnit` on all five of Metal/CUDA/OpenCL/GLSL/WGSL (no object
representation at all — C's `void` is not a value, and WGSL has no unit type),
plus WGSL's `TBool` (a real value the language will not let into a buffer;
`naga` refuses it rather than picking a width).
`test_no_memory_form_set_is_exactly_as_recorded` fails on
any addition or removal, so widening the exemption is a deliberate edit to a
literal list rather than something a codegen change can do quietly. A model
whose admissible-outcome count is misreported is a model of something else, so
this is stated here in the same terms as the code.

That is exactly "the exhaustive table settled in an afternoon", and it found the
two the seam was still hiding on the device side — Metal's
`TFloat64 -> "float"` (8 bytes read at 4) and `TBool -> "bool"` (4 read at 1).

**What this is and is not.** It is proof by exhaustive cases over a finite
domain, which is the same argument a Rocq model of a total function would make;
the difference is only *when* it is checked — `dune runtest` rather than
`rocq check` — and that the enumeration's totality is enforced by OCaml's
exhaustiveness checker rather than by `Coq`'s. For a function with no theorems
worth stating, that trade looks right, and it is the cheapest available reading
of this lesson. It does not extend to anything with an interesting invariant.

**A caveat this entry should carry, because backlog-141 supplied it.** The
front-half validator had been green since it was written while **not sweeping
two of its ten members**: its `unfold` pushed the successor instead of the
element just visited, dropping the first of each chain and duplicating the last,
and its own anti-vacuity length check could not see this because one duplicate
exactly compensated for one omission. A cheap model is only worth its line count
if the enumeration it rests on is *checked to be the enumeration it claims* —
both files now assert distinctness as well as length. Machine-checked totality
is the thing Rocq would have given for free here, and is the honest argument in
its favour.

### M-02 — the silently-succeeding-wildcard family, out of scope

| | |
|---|---|
| Date | 2026-07 |
| Instances | 4 in a single day |
| Tracked as | audit 2026-07-24 items backlog-48, backlog-49; backlog-94 |

Backend intrinsic dispatch emitting a raw name and returning `Ok` for an unknown
intrinsic, producing invalid device code.

**Why it was missed.** No formal model covers backend code emission at all. This
is a miss by scope, not by weakness, and is recorded so that the scope boundary is
a stated fact rather than something inferred later from the absence of findings.
The defence that landed for this class was an ordinary Alcotest
(`test_intrinsic_fallback_all.ml`) and a dispatcher refactor, not a proof — which
is the right answer, and is itself a data point about where formalisation is *not*
the cheapest instrument.

### M-03 — the apparatus did not notice its own unverified bridge

| | |
|---|---|
| Date | 2026-07-24 (audit), fixed 2026-07-27 (backlog-46) |

`test_layout_conformance.ml` checked the production layout module against a
130-line **hand transcription** of `PtxLayout.v`. Every theorem proved about the
Rocq definitions was then applied to a copy that nothing compared to the original.
The formalisation had a hand-maintained hop between the proofs and the code under
test for the whole of its existence, and it took a general code audit — not any
part of the formal apparatus — to notice.

**Lesson.** The chain is only as strong as its least-checked link, and the
least-checked link was inside the verification apparatus rather than in the code
it verified. Worth remembering when reading any of the catches above: they are
claims about a chain that had an unchecked link in it until 2026-07-27.

## Null results

<a id="n-01"></a>

### N-01 — replacing the hand mirror found no defect

| | |
|---|---|
| Date | 2026-07-27 |
| Work | backlog-46 — extract `PtxLayout.v` to OCaml, drop the transcription |
| Defects found | **0** |

When the 130-line hand transcription was replaced by Rocq's own extraction, the
conformance suite passed **11/11 unchanged**. The transcription was faithful.

**Recorded as a null result, deliberately.** The work bought a structural
guarantee — the transcription can no longer drift, because there is no
transcription — and it bought no bug. Both halves are true and the second is the
one that will be forgotten. If the next three pieces of verification-infrastructure
work also return zero, that is a pattern this file exists to make visible rather
than absorb.

### N-02 — reconciling the proof ledgers found no proof defect

| | |
|---|---|
| Date | 2026-07-27 |
| Work | backlog-95 — generate the ledger, enforce the axiom allowlist |
| Spec/impl defects found | **0** |
| Apparatus defects found | 4 of 5 count figures wrong, 1 phantom anchor |

Every theorem the three hand-written ledgers claimed was proven, and that exists,
is proven. What was wrong was the bookkeeping. Measured against the toolchain:

| file | claimed | actual |
|---|---|---|
| `convergence-safety/proof-ledger.json` | 59 proven, 19 definitions | 111, 49 |
| `convergence-safety/proof-ledger/proof-ledger.json` | 68 proven, 18 definitions | 111, 49 |
| `type-safety/proof-ledger/proof-ledger.json` | 90 proven, 36 definitions | 90 ✓, 87 |
| `codegen-ptx` | *(no ledger at all)* | 79, 60, 6 axioms |

Four of the five figures were wrong, the worst understating its project by 52
theorems; the single correct one was type-safety's theorem count. And one entry —
`check_env_nonvarying_uniform`, status `PROVEN` — named no theorem at all; it was
an anchor invented for a *pair* of real lemmas.

**Recorded here rather than as a catch** because none of it is a defect in the
compiler or in a proof. It is a defect in what the repository *said* about its
proofs, which is a distinct and lesser thing. It does bear on every claim in this
file: until 2026-07-27 the counts quoted in `STATUS.md`, in commit messages and in
the ledgers were not derived from the toolchain, so any of them cited as evidence
of coverage was citing a number nobody had checked.

---

## How to add a row

Append to the relevant section when any of these happens:

- a proof fails to close and the reason is a defect → **Catches**
- a conformance/differential run produces a counterexample → **Catches**
- extending a model reveals the implementation disagrees → **Catches**
- a defect ships in an area a model covers → **Misses**, with the specific reason
  the model did not reach it
- a piece of formal work completes having found nothing → **Null results**

State the counterfactual. If the honest counterfactual is "review would have found
this", write that.
