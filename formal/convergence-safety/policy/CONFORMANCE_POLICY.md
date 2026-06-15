# Conformance Policy — {Project} Formal Component

**Standing policy** (2026-04-21; generic core extracted 2026-06-11,
apparatus v1.1 — locked semantics unchanged). This file is the single
authority on what qualifies as a conformance test in this project.

Placeholders in `{braces}` are resolved per host/project from the
profile (`profiles/`) at `/formal-init`. This policy's generic core was
extracted from the tezos/Octez staking project's v1.0 policy; the
staking worked examples and host-specific material it cited inline are
archived in `examples/tezos/conformance_examples.md` (cited below as
"the examples archive").

## The one rule

**A conformance test compares the extracted certified model (produced by
Rocq `Extraction` from the spec) against the live target implementation
in `{protocol_src}`.**

Nothing else counts.  A test that compares the model to a hand-computed
integer, to a direct OCaml formula, or to itself is NOT a conformance
test — it is a *model check*.

## Corollaries

### Label immunity

Every label used in this project — **conformance, differential, attack-mode,
boundary-probe, fuzz, diagnostic** — is a synonym for conformance in the
context of this project.  A test that wears any of these labels MUST
satisfy the one rule.  If it doesn't, it's mislabelled and belongs in
`model_checks/`.

### Directory discipline

Two sibling directories, each with a defined role and enforcement.

| Directory | What it contains | Enforcement |
|---|---|---|
| `test/` | Conformance tests only | Every `.ml` executable must import and call at least one `{protocol_lib}` symbol (see list below). |
| `model_checks/` | Legacy model-only + spec-only tests | Executables must NOT call `{protocol_lib}` symbols (they may link the shared helper library for convenience, but use only pure-model functions from it). |
| `test_helpers/` | Shared utilities (machines, generators, coverage probe) | Links `{protocol_lib}`.  Used by both of the above.  Not itself a test directory. |

**New files under `model_checks/` are forbidden going forward.**  The
directory exists for historical continuity; every new test lands in `test/`.

### Tool discipline

Every new conformance test is PBT-driven, using the apparatus's locked
PBT stack (see `policy/PBT_STACK.md`):

- **QCheck2** — quick tier (suite-default / pre-push)
- **Monolith-native** (F. Pottier's Monolith in random-generator mode) —
  long tier (nightly / on-demand)

A host-standard PBT framework declared in the project profile may be
admitted in addition (tezos: bam / tezt-bam); Crowbar is reserved for
arithmetic-unit fuzzing of pure functions, not stateful workloads.  The
v1.0 staking-era four-framework roster is archived in the examples
archive.

Hand-written Alcotest scripts are **not a valid tool** for new coverage.
Alcotest is acceptable as the *runner* of a PBT-driven test (e.g. via
`QCheck_alcotest.to_alcotest`), but the body must be PBT-driven.

**Exception** — regression tests for specific historic bugs may be
hand-written Alcotest cases living under `test/regressions/`.  These are
still conformance (they call `{protocol_lib}`), just lower-effort than
PBT.  Each regression file starts with a comment linking to the original
finding in the project's findings record.

### Exploratory screens must run the proto code being formalized

Exploratory screens and sweeps (anything under `experiments/`, and any
ad-hoc triage harness) must step the **real target implementation** or
the **certified extraction** (`{Project}Model.*` transition functions)
— never a hand-reimplemented transition.  A local `tick`/`step`/
`proto_tick` that recomputes the step by hand with no proto/extraction
call produces numbers with no evidentiary value: when the
reimplementation diverges from the target, every conclusion drawn from
the screen points the work in the wrong direction.

Screens are triage; validation is Rocq + conformance (extraction ≡
target).  A screen that fails this rule is quarantined — its numbers
are disregarded until it is re-pointed at the target or the extraction
(`/formal-check` rule 9 / Check 10 enforces this; see
`docs/activation.md`).  Origin: SWRR continuations 73–74, where 33
reimplementation screens disagreed with the target and misdirected an
attack for two sessions (`memory/feedback_screens_must_run_proto.md`).

## What counts as a `{protocol_lib}` symbol

For a test to qualify as conformance, its body (not imports) must
include at least one symbol from the project's qualifying list
`{protocol_symbols}`, directly or through a helper that does.  The
qualifying list is declared per project (profile + project `STATUS.md`);
the staking project's tezos list is archived in the examples archive as
a worked example.

A test that only uses `{Project}Model.*` is a model check, not conformance.

## Supervisor enforcement

`/formal-check` Step 2b scans every `.ml` executable in `test/` for at
least one of the qualifying symbols.  Absent → `CONFORMANCE ALERT` →
forces **Pivot** recommendation (never Continue).  The reverse check on
`model_checks/` flags any `{protocol_lib}` symbol usage → forces Pivot
(move the file to `test/` or remove the call).

"Step" labels (2b, 2c, 2d, 3, 3b, 3c, …) are this policy's internal
names for its enforcement steps; the apparatus-level audit that runs
them is `/formal-check` (see `docs/activation.md §/formal-check`).

## Failure triage (v3, 2026-04-21)

Rationale: strong confidence in `target ≡ certified model` requires
every divergence to terminate in a concrete fix on one specific side.
Silent workarounds erode confidence; tracked divergences preserve it.

### Model check fails (`model_checks/`)

Root cause is always in the Rocq layer.

1. Verify generator inputs are legal. If not, the **generator** is the
   bug — fix it.  (Correcting test infrastructure is not "hiding a
   failure".)
2. If inputs are legal, the spec/model/extraction is wrong.  Fix at
   Rocq, re-extract, re-run.
3. Record as a Finding only if the bug was non-obvious or cost more
   than one session.

**Gates are forbidden in `model_checks/`.**  No `return None` that
skips a legal-input failure.

### Conformance test fails (`test/`) — two-phase triage

**Phase 1 — rule out harness bugs first (autonomous).**

Conformance failures are spec-vs-target *only after* the harness is
proven faithful.  Check:
- Generator inputs legal under both spec and target preconditions.
- Machine pre-processing faithful to the spec's input contract.
  (Origin: staking Findings 2 & 3 — a doubled conversion in the
  harness; see the examples archive.)
- Projection function compares the right fields.
- **Fixture parameters synced on both sides.** When a fixture changes a
  model parameter, the corresponding target-side constant must be
  changed too.  Changing only the model side leaves the target on its
  default and produces a harness-internal divergence that masquerades as
  (a')/(a)/(b).  (Origin: staking Mandate A.1, 2026-04-22, hit this
  exact trap; worked example in the examples archive.)
- **Proto-as-oracle scope.** The harness may query the target for two
  purposes only: (i) **boundary seeding** — at `build()` time, project
  target state into the initial model state so subsequent transitions
  run off the model; (ii) **projection / lockstep** — observe the
  target's current state to compare against the model's current state.
  The harness must **NOT** query the target to compute, substitute, or
  pre-correct values the model is responsible for at runtime.
  Symptom: code in the harness that reads target data AND writes a
  derived value into the model state mid-scenario (rather than at
  `build()`) is a likely violation.  Closure path: identify the
  missing model field, lift it into the model, retract the query.
  If lifting is genuinely deferred, document as a
  `STATUS.md §"Trust root"` entry with `Remove when:` predicate —
  do not silently retain the proto-as-oracle.  (Origin: staking
  session-34 audit, 2026-04-27; the Phase 6.6.5 worked example is in
  the examples archive.)

If any is "no", classify as **(c) harness**, fix autonomously, rerun.
(c) is recorded as a Finding but is **not** a conformance divergence.

**Claim-must-cite-test rule (standing guardrail, from staking
Finding 5, 2026-04-22).**  Comments in `theories/*.v`,
`test_helpers/*.ml`, or `extraction/*.ml` that assert a specific
target behaviour (`"proto does X on this path"`, `"matches the
implementation's semantics where ..."`, etc.) **MUST cite the test
that exercises that path**.  A citation looks like
`(* verified by test_{project}_proto_conformance TC_<N> *)` or
`(* see test/regressions/test_{project}_finding_N.ml *)`.

Uncited claims are **un-verifiable assertions about the target** —
exactly the class of error that surfaced the originating finding (an
uncited behavioural comment mis-claimed the target's behaviour with no
test to catch the mistake; it was discovered accidentally by an
unrelated scenario one session later).  `/formal-check` should surface
uncited target-behaviour claims as warnings (Step 2f — future
enforcement addition).

**Bridge-layer port discipline (standing guardrail, from staking
session-34 audit, 2026-04-27).**  Hand-written OCaml under
`extraction/` whose docstring claims "mirror of `theories/X.v`" /
"line-by-line port" / "equivalent to the spec by inspection" is the
**claim-must-cite-test antipattern at the spec↔OCaml boundary** — the
equivalence is asserted but unverifiable.

Every OCaml file under `extraction/` whose body implements something
defined in `theories/` must satisfy at least one of:

- **Auto-generated** by `Rocq Extraction` from the spec
  (`Extraction "{Project}ModelZ" symbol1 symbol2 ...` in the project's
  extraction theory), with a thin int-facing facade if the harness
  wants labelled args / OCaml-int indices (the `{Project}ModelZ` →
  `{Project}Model` pattern).  This is the preferred route —
  equivalence is by construction.
- **Bound to the spec by a Qed Conforms lemma** in a sibling
  `theories/*Conforms.v` file.  The hand-written OCaml's docstring
  must cite the conformance theorem
  (e.g. `(* Spec equivalence: see <Theory>Conforms.v <lemma_name> *)`).

Hand-written OCaml without either route is an open trust gap and
must be tracked as a TODO + `Remove when:` predicate in
`STATUS.md §"Trust root"`.  `/formal-check` Step 2h (future
enforcement) should flag any `extraction/*.ml` file whose docstring
contains "mirror"/"port"/"equivalent" without a `Conforms.v`
citation or a `Rocq Extraction` provenance line.  (The staking
artifacts this rule was extracted from are listed in the examples
archive.)

**Phase 2 — classify spec-vs-target divergence (user-gated).**

The supervisor (Claude) produces the diagnosis.  **The user classifies.**
Under no circumstances does the supervisor install a gate, fix the
spec, file an upstream change, or mark a test as expected-failure before
the user has reviewed the filled Finding template.

**Phase 2 trigger — surfaced OR analytically discovered.**  A
spec-vs-target divergence triggers this workflow whether it was
surfaced by a failing test OR discovered by *analysis* during scope-
bounded implementation (e.g. an implementer notices a divergent edge
while reading the target source).  Documenting the edge in a code
comment or commit message is **not** a resolution — it is the
uncited-claim antipattern (see §Phase 1 claim-must-cite-test rule)
by another name.  The policy-mandated sequence below runs for
analytically-discovered divergences identically to test-surfaced
ones; the reproducing regression is the artefact that moves the
claim from uncited to cited.  (Rule added from staking Mandate A.2,
2026-04-22: a divergent edge was "documented but not filed" before
this rule was landed.)

Workflow:

1. Reproduce as a fixed scenario under `test/regressions/`
   (hand-written Alcotest, per the Exception in the Tool discipline
   section above).
2. Side-by-side the spec (Rocq + LaTeX) against the target source.
3. Fill the Finding template (see below): Symptom, Side-by-side,
   proposed classification *or* UNCLASSIFIED with reasoning.  Leave
   Gate / Upstream / Resolution blank.
4. **Stop.  Alert the user.  Present the filled template.**
5. User decides: classification + whether a gate is acceptable + next
   action (fix Rocq, file upstream change, gather more data, consult
   team).
6. Only then does the supervisor execute.

Categories, reference = the LaTeX spec (`{Project}Spec.tex`) when
reviewed, else the documented behavior from the prior version /
governing specification:

- **(a) spec wrong** — Rocq+LaTeX formalize the wrong thing.  Fix:
  update LaTeX + Rocq + re-extract + re-prove refinement.  LaTeX
  update is mandatory (enforced by `/formal-check` Step 3) and must
  conform to Resolution-quality rule 3 (self-contained spec; no
  finding / session / mirror-target refs).
- **(a') spec imprecise** — spec is silent; the target picks a
  resolution.  Fix: extend LaTeX + Rocq to specify.  Same
  Resolution-quality rule 3 constraint on the LaTeX shape.
- **(b) target wrong** — the implementation deviates from intended
  semantics.  Fix: file the upstream change; Finding stays OPEN with
  `Upstream:` link; regression test stays in place.

### Resolution-quality rules

Three rules governing how a Finding is closed, accumulated from
staking Findings 5–6 (2026-04-22/23) and session 59 (2026-05-06).

**1. Fix-first, document-last (default preference).**
Documentation-only resolutions (entry under §"Acknowledged bridge
gaps" with `Remove when:` predicate) are **last resort**, acceptable
only when:
- a spec fix (extend LaTeX + Rocq) is provably impossible without
  team expertise or external input (e.g. governance question about
  intended semantics), OR
- a spec fix was attempted and rejected (either too invasive
  relative to value, or introduces a deeper divergence), with the
  rejection documented in the Finding template.

The default resolution for (a)/(a') is **fix the spec**.  If the
implementer proposes an acknowledged-gap resolution, the Finding
template must include a `Why not fix:` paragraph.  Rule added from
staking Finding 6 (2026-04-22): the initial recommendation was
acknowledged-gap; the user overrode to spec extension, which closed
the divergence properly.

**2. Scope-restriction retraction on closure.**  When a Finding's
resolution removes the *cause* of an earlier scope restriction
(e.g. "X is NOT in the generator because of divergence Y"), the
resolution commit (or a prompt follow-up) **must** either (i) retract
the restriction (re-enable the scope-restricted code), or (ii)
explicitly document why the restriction persists post-fix.  Leaving
an obsolete scope-restriction comment in place re-creates the
uncited-claim antipattern: the code carries a justification that no
longer holds.  Rule added from staking Finding 6 closure audit
(2026-04-23), where an obsolete generator restriction blocked
random-PBT validation of the spec extension it had been waiting for.

**3. LaTeX reads as a self-contained specification.**  The paper
specification (`{Project}Spec.tex`) describes *what the spec is*, not
*how it got there* nor *how it tracks the implementation*.  Every
LaTeX edit landed by a Finding closure must conform to all of:

  - **No finding / session / amendment refs.**  No `Finding N`,
    `Mandate X.Y`, phase/stage labels, `(YYYY-MM-DD)`,
    `(session N)`, `RESOLVED` / `Remove when:` in the rendered text.
    Closure history belongs in `STATUS.md`, `history/JOURNAL.md`, and the
    commit message — not in the spec.
  - **No "mirroring the implementation" / "matches the target's X"
    explanations.**  The LaTeX defines spec semantics directly.
    Justifications of the form "the spec uses ceil because the
    implementation does so" are development commentary, not
    specification content.
  - **No implementation file/line pointers.**  No `<file>.ml:30-34`
    refs, and no target-internal identifiers used purely to point at
    the implementation (module-qualified function names, error-
    constructor names, etc.).  The exception is the spec's
    *parametric* constant names — when the LaTeX names a
    section-variable (e.g.\ "$P \in (0, 10000]$
    `max_slashing_per_block`"), the implementation-side name
    identifies *what the variable is*, not *how the spec mirrors it*.
  - **No Resolved-Questions / Q&A artifact sections** ("Resolved",
    "Open questions", development trace, etc.).  Open questions
    belong in `history/JOURNAL.md`; resolved ones stop being mentioned.
  - **Forward references to the mechanisation are allowed, but
    sparingly.**  A pointer like "lemma `<lemma_name>` in
    `theories/<Theory>.v` establishes the spec--implementation
    equivalence" is acceptable when it tells the reader *where the
    formal proof lives*.  It is not acceptable when used to
    motivate the spec's design (which is forbidden by the previous
    bullets).

Rule added 2026-05-06 (staking session 59) after a post-closure
cleanup of the staking LaTeX: the document had accumulated dozens of
finding annotations, "mirroring proto's `Foo_bar.baz`" prose, and
file-line refs across multiple sessions, which made it read as a
development log rather than a specification.  The cleanup removed
~150 lines without losing any spec content.

The pattern: every Finding closure that touches LaTeX is tempted to
add a "Finding N closure (session M)" paragraph "for traceability".
Resist.  Traceability lives in version control, in `STATUS.md` /
`history/JOURNAL.md`, and in the commit message — not in the spec.

### Gate lifecycle

A gate is a temporary bridge while an (a)/(a') Rocq fix is in flight.
Every gate must:

1. Be tagged **inline** with `Finding N` + a one-line reason.
2. Have a `Remove when:` predicate in the Finding entry.
3. Be listed in `STATUS.md`'s **Known gates** section.
4. Be re-surfaced by `/formal-check` Step 2d on every invocation until
   closed.

Gates for (b) are NOT permitted — if the target is wrong, the
regression must stay visible.  Gates are NEVER installed autonomously
by the supervisor — only on explicit user approval after the filled
template has been reviewed.

### UNCLASSIFIED findings

If Phase 2's side-by-side does not yield a confident (a/a'/b):

- `Classification: UNCLASSIFIED` in the Finding.
- The failing scenario stays in `test/regressions/` as a **red** test
  (no gate, no expected-failure marker).  The test suite reports the
  failure on every run until the user classifies.
- The supervisor must not apply any workaround.  Next-step decision
  is the user's.

### Finding template (standardized)

```
**Finding N — <title> (<STATUS> <date>).**
- Symptom: <one-liner>
- Classification: (a) | (a') | (b) | (c) | UNCLASSIFIED
- Side-by-side:
    - Spec:   <Rocq fn + location, paraphrased semantics>
    - Target: <file + fn + location, paraphrased semantics>
- Regression: test/regressions/test_<project>_finding_N.ml
- Gate: <inline location>, remove when: <predicate>   (blank until user approves)
- Upstream: <change link, if (b)>                     (blank until user files)
- Resolution: <fix description + commit, filled on close>
```

### Enforcement additions to `/formal-check`

- **Step 2c** — scan `model_checks/*.ml` for `return None`-style gates;
  warn on any.
- **Step 2d** — scan `test/*.ml` and `test_helpers/*.ml` for
  `Finding N` markers; cross-check against `STATUS.md` Known gates and
  open Finding entries; warn on orphans (gate with no Finding, Finding
  with no gate, closed Finding with leftover gate).
- **Step 3b** — for every OPEN (a)/(a') Finding, verify LaTeX was
  touched alongside Rocq in the fixing commit.
- **Step 3c** — scan the rendered LaTeX (`pdftotext {Project}Spec.pdf -
  | grep -iE ...`) for the forbidden tokens enumerated in
  Resolution-quality rule 3.  Warn on any match.  The generic
  greppable blocklist is:
  ```
  mirror | proto's | the implementation's | Mandate | Finding [0-9]
  | session [0-9] | [0-9]{4}-[0-9]{2}-[0-9]{2} | Phase [0-9]
  | Stage [A-Z] | \.ml:[0-9] | Resolved Questions
  ```
  extended per project with target-implementation identifier tokens
  (module names, error constructors) declared in the profile — the
  staking/tezos token list is archived in the examples archive.
  Section headers and parametric-constant name labels in
  `\noindent\textbf{...}` or definition itemise are exempt because
  they identify the parametric constants the spec is over, not
  mirror commentary.

### Dual-axis coverage (2026-04-23)

**Coverage has two orthogonal axes and BOTH must be measured + reported**
when comparing tools or evaluating coverage deltas from a generator /
decoder / fixture change:

1. **Domain-space coverage** — `test_helpers/coverage_probe.ml`'s
   eq-class hash over the project's coverage dimensions.  Answers
   "which edge tuples did the walker reach?"  Fast, built-in, tracked
   by the project's bakeoff executables.  Metric: `eq_classes`.
2. **Source-code coverage** — `bisect_ppx` line/branch coverage of
   the target modules in `{protocol_src}`.  Answers "which
   implementation lines executed?"  Authoritative for reviewers.
   Metric: per-file `covered / total` from
   `bisect-ppx-report summary --per-file`.

The two axes can move independently: a single staking fixture change
raised eq_classes 98 → 177 (domain) AND raised one target file's
source coverage 15.85 → 70.49 % (see the examples archive).  A
coverage claim that only reports one axis is incomplete.

**Mandatory reporting format** for any generator-level coverage change:

```
Before: eq_classes=X / target-file coverage ∈ [Y%, Z%]
After:  eq_classes=X' / target-file coverage ∈ [Y'%, Z'%]
        (list per-file deltas > 5 pp)
```

**Recipe.**  See `report/BENCHMARKS.md` §"Source-code coverage via
`bisect_ppx`" for the project's local measurement workflow.  Host-CI
coverage runs (when the host repo provides them) are canonical for
published numbers; the host commands are declared in the profile.

## Policy-accrual discipline (meta-rule)

**After every new situation not covered by an existing policy rule,
pause and ask: should this be enforced by policy?**

The rationale is workflow automation: the policy should accrue rules
derived from real cases, so future agents follow the accumulated
discipline without having to re-derive it each time.  Each iteration
that adds a rule makes the next iteration faster and less drift-
prone.

Concretely, after resolving any new kind of situation (novel Finding
class, novel mandate outcome, novel harness / generator pattern, etc.),
the implementer answers one of:

- **Yes, codify** — land a one-paragraph rule in the relevant
  `policy/CONFORMANCE_POLICY.md` section, ideally in the same session or as
  an immediate follow-up commit.  Cite the triggering case so the
  rule's origin is traceable (e.g. *"(from Finding 5, 2026-04-22)"*).
- **No, context-specific** — briefly note in the session journal why
  the case doesn't generalise.  Prevents future agents from re-asking
  the same question.

Origin examples landed via this rule in the staking project (v1.0):
- Fixture-sync rule (§Phase 1, from Mandate A.1).
- Claim-must-cite-test (§Phase 1, from Finding 5).
- Latent-divergence rule (§Phase 2, from Mandate A.2).
- This meta-rule itself (from the Mandate A.2 reflection).
- Bridge-layer port discipline (§Phase 1, from the session-34 audit).
- Proto-as-oracle scope (§Phase 1, from the session-34 audit).
- Policy-aligned next-step proposals (§Next-step discipline below,
  from the session-38 reflection).

A `/formal-check` Step 2g (future) may audit the journal for
"new situation → no policy addition" entries and require the
implementer to name the reason, closing the loop.

### Next-step discipline (2026-05-01)

**When the user asks "what's next?" or the supervisor proposes a next
step, the proposal MUST be ordered against the policies below before
any options are presented.**

Triggering case (staking session 38): the supervisor's initial
"what's next" framing surfaced four roughly co-equal options without
weighing them against the open-findings table.  The user asked "what
do our policies and workflow say?", which forced a re-derivation.
Codifying prevents the same drift in future sessions.

**The check, in order:**

1. **Open findings table** (`STATUS.md §Open findings`).  Any OPEN
   `(a)`/`(a')` finding has higher priority than discretionary work.
   The supervisor presents the *Remove when:* predicate as the next
   step, not as one option among others.  Resolution-quality rule 1
   (fix-first, document-last) defaults to spec fix; acknowledged-gap
   requires explicit user override with `Why not fix:` paragraph.
2. **Known gates** (`STATUS.md §Known gates`).  Active gates are
   tracked workarounds awaiting their `Remove when:` predicate.  An
   open gate has higher priority than coverage / benchmark work.
3. **Phase 2 triggers** (§Phase 2 trigger above).  If recent work
   surfaced a spec-vs-target divergence (analytically or by failing
   test), the Phase 2 workflow runs to completion before any
   discretionary work resumes.  The supervisor must not propose
   coverage / benchmark / new-area work that bypasses an unresolved
   Phase 2.
4. **Sequencing guidance** (`findings/UNCOVERED_EDGE_CASES.md §Suggested
   sequencing`, `STATUS.md §Phase roadmap`).  Within "no open finding"
   territory, follow the documented sequencing.  Coverage tiers and
   bakeoff refreshes are interleaved per findings/UNCOVERED_EDGE_CASES.md, not
   sequential blocks.
5. **Discretionary work last.**  Push to remote, merge to target
   branch, start a new project area — these come AFTER 1–4 are
   clean.  The supervisor may surface them as parallel options once
   the policy-mandated steps are complete or scheduled.

**Output shape.**  When proposing next steps, the supervisor MUST:

- Cite which policy section governs the proposal (e.g. *"Per §Open
  findings, Finding N is OPEN with `Remove when:` = …; per
  Resolution-quality rule 1 the default is spec fix"*).
- Order proposals by policy priority, not by appeal or estimated
  effort.
- Flag any policy-discretionary option as such, distinct from
  policy-mandated steps.

**What this rule rejects.**  A "menu of options" framing where the
supervisor lists policy-mandated and discretionary work side-by-side
without surfacing the difference.  The user retains final discretion;
the supervisor's job is to make the policy-mandated steps visible.

## Acknowledged bridge gaps

Findings resolved as permanent design records — via Resolution-quality
rule 1's last-resort route, or an (a') classification resolved as
acknowledged design — are retained in this section of the project's
policy copy as **design records**, not open gaps.  Each entry records:
what the gap was/is, why it isn't closable by proof alone, the existing
acknowledgment at the test level, the user-set classification, and the
resolution.  `/formal-check` Step 2e (future) may omit (a')-resolved
entries from its "orphan gap" warning list.

The staking project's two entries (`shares_of_tez_conforms` cap-active
regime, CLOSED by a reachability proof; `slash_pct_da_bounded` 1-unit
rounding gap, resolved as acknowledged permanent design) are archived
as worked examples in the examples archive.

## History

This policy was made explicit after a multi-session drift in the
staking project where several tests claimed the "conformance" label
without going through the target implementation (four tests renamed /
moved / split; the inventory is in the examples archive).

Root causes of the drift (retained here so future maintainers see the
pattern):

- Inherited names (`test_<project>_conformance.ml`) were not challenged.
- PBT-framework vocabulary ("differential testing") was imported
  without filtering through this project's policy.
- The supervisor's Step 2b used `dune`-global checks, not per-file
  checks.
- Memory notes existed but were not promoted to a gated policy file at
  the project root — making them easy to reinterpret session-by-session.

This file closes all four gaps.
