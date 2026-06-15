# Divergence Audit Policy — {Project} Formal Component

**Standing policy** (2026-05-08; generic core extracted 2026-06-11,
apparatus v1.1 — locked semantics unchanged).  Sister document to
`policy/CONFORMANCE_POLICY.md`.  This file is the single authority on
the methodology and quality bar for divergence findings against the
project's formal spec.

`policy/CONFORMANCE_POLICY.md` governs **spec ↔ target conformance**
(the extracted model matches the target implementation).  THIS policy
governs **audit findings** (gaps, attack surfaces, missing model
coverage) identified by the three-lens audit framework, and the
quality bar for closing them.  Both apply concurrently: a divergence
closure must satisfy both this policy and the relevant conformance
requirements.

The audit document this policy governs is the project's
`findings/DIVERGENCE_FINDINGS.md`.  The staking-project worked examples
the v1.0 policy carried inline are archived in
`examples/tezos/divergence_examples.md` (cited below as "the examples
archive").

## The atomicity rule

**A divergence finding is closed only when ALL seven components land in
ONE atomic unit, before the next finding is touched:**

1. **Spec patch** — theory edit at the right architectural layer
   (per-component / network / extraction).
2. **Wf / model_wf / sim lemmas at Qed** — every new transition gets
   its `_wf` and `_model_wf` lemmas; every new field's structural
   invariants are proven; **the appropriate simulation record MUST be
   extended with a sim field AND the corresponding refine line in the
   simulation proof for every new T-numbered transition** (codified
   2026-05-08 from a staking retroactive gap — two closures originally
   added wf lemmas but missed the simulation-record slot, surfaced
   only by the post-policy walk; see the examples archive).  Zero
   `Admitted`, zero axioms preserved.
3. **Extraction** — `make -f Makefile.coq` regenerates
   `extraction/{Project}ModelZ.{ml,mli}` cleanly; the OCaml facade
   wrapper in `extraction/{Project}Model.ml` is added or updated for
   every new transition or helper.
4. **Differential test** — new regression
   `test/regressions/test_{project}_f<NN>_<short_name>.ml` covering
   pure-spec correctness AND a target-vs-spec lockstep assertion when
   the harness can construct it.  Listed in
   `test/regressions/dune` with a comment header citing the finding.
5. **LaTeX paper update** — `{Project}Spec.tex` updated when the
   finding touches spec content (state record, transition definition,
   helper).  `pdflatex` compiles cleanly.  No F-NN references in the
   paper (per §Resolution-quality rule 3 below).
6. **Full test suite GREEN** — all regressions + the project's
   proto-conformance suite + its PBT suites.  Spot-checking adjacent
   regressions is NOT sufficient.
7. **Doc updates** — the `findings/DIVERGENCE_FINDINGS.md` finding's
   `**Status**` field flips to `**CLOSED YYYY-MM-DD**` with closure
   record (modules touched, lemmas added, tests added,
   regression-count delta, any documented limitations); `STATUS.md`
   `## Divergence audit` section gets a per-closure entry.

### Anti-pattern smell

If at any point during a closure you find yourself thinking "I'll
batch the LaTeX / proof / full-suite for later" — STOP.  Land it now
or open a fresh task before claiming the finding closed.  Closures
that touch only `theories/` and `test/regressions/` while skipping
`{Project}Spec.tex` or the wf-lemma section of the same theory file are
the 2-phase pattern by another name.

### Why atomicity

- **Bisect / revert**: each closure is a single coherent unit.
  Reverting one finding doesn't entangle the others.
- **No accumulated debt**: closure quality stays uniform; the auditor
  doesn't have to remember which findings still need LaTeX or
  proofs.
- **Fresh context**: lemma proofs and LaTeX wording are easier to
  write while the spec change is in working memory, not days later.
- **Mirrors UNCOVERED**: edge-case closures
  (`findings/UNCOVERED_EDGE_CASES.md`) are each fully landed before
  the next.  Divergence findings follow the same discipline.

## Three-lens framework

Every audit walks the in-scope module set in this order:

1. **Lens 3 — abstraction-boundary** (mechanical, first): walk every
   target module in scope, mark every branch the spec doesn't model.
   Produces a finding-per-gap enumeration.  Catches what coverage-
   style audits miss because conformance can't exercise gaps the
   spec doesn't represent.
2. **Lens 1 — scenario-model** (creative, seeded by lens-3): pick
   scenarios + adversarial goals; re-rate each lens-3 finding's impact
   by scenario reachability.  Lens-3 alone can't judge reachability
   without the reachability model in hand; lens-1 can't enumerate without
   lens-3's coverage map.
3. **Lens 2 — composition** (impact-only triage, last): cross-
   module attack paths that compose two or more lens-3 / lens-1
   findings.  Impact-only entries grouped by scenario goals; full-
   finding format reserved for lens 3.

Lens-1 is **policy-mandated** as a sub-phase before any audit step
declares closure (per
`memory/feedback_next_step_discipline.md`).  Skipping lens 1 is a
policy violation.

## Coverage taxonomy

Defined in the project's `findings/DIVERGENCE_FINDINGS.md` §Coverage
taxonomy: a set of scenario profiles (who the adversary/actor is), a
set of G-numbered adversarial goals (what they gain), and a five-level
impact scale (CRITICAL / HIGH / MEDIUM / LOW / INFO).  Impact reflects
scenario-reachable consequence under realistic capabilities, NOT
worst-case in isolation.  The staking instantiation (five scenario
profiles, seven monetary goals G1–G7) is archived as a worked example.

## Finding template

Every finding in `findings/DIVERGENCE_FINDINGS.md` uses this structure:

```
### F-NN — <short title>

- **Module**: `{protocol_src}/<file>.ml:<line>`
- **Spec coverage**: <which spec transition should model this, or "unmodelled">
- **Branch**: <the target branch / state mutation that the spec doesn't reach>
- **Lens**: 3 (abstraction-boundary)
- **Scenario reach** *(provisional, lens 1 will refine)*: <which scenario(s) and which goal(s) this gap could serve, or "not reachable" for INFO findings>
- **Lens-1 reach** *(YYYY-MM-DD)*: <scenario profile + cost + harness-vs-production split + top-N candidacy>
- **Impact**: CRITICAL / HIGH / MEDIUM / LOW / INFO (lens-1; was X per lens-3 if changed)
- **Mitigation proposal**: <spec extension / harness extension / target patch / out-of-scope justification>
- **Status**: open / triaged / closed / out-of-scope / deferred (with rationale)
```

After lens-3 enumeration completes, the lens-1 walk re-rates each
finding's impact based on scenario reachability.  Impact
provenance is preserved via `(lens-1; was X per lens-3)` annotation
when a lens-1 verdict changes the rating.

## Status enum

Five states, of which **four are terminal** for settled-milestone
purposes:

- **open** *(non-terminal)* — finding stands; closure work pending.
  An "open" finding has had its lens-1 reach rated but no closure
  path landed and no explicit deferral rationale.  Open findings
  block audit settling.
- **triaged** *(terminal)* — finding rated LOW or INFO with explicit
  documented rationale (typically: "not reachable for any G-goal
  under the coverage taxonomy" or "track-and-watch — depends on future
  target change").  Triage rationale lives in the finding body
  alongside the lens-1 verdict.
- **closed** *(terminal)* — closure landed atomically per §The
  atomicity rule.
- **out-of-scope** *(terminal)* — finding has its root cause in
  another formal project; recorded in the appropriate out-of-scope
  cluster section (per §Out-of-scope cluster pattern below).
- **deferred-with-rationale** *(terminal)* — finding is real and
  reachable but closure depends on infrastructure not yet built
  (e.g.\ harness extension, cross-domain bridge spec); rationale
  must cite the prerequisite and a `Remove when:` condition.

An audit phase / overall audit only settles when every finding is
in a terminal state — i.e.\ no `open` entries remain.  The
settled-milestone sweep at audit closure flips legitimate "open"
findings to the appropriate terminal state (most often `triaged`
when the lens-1 verdict shows no scenario reach, or
`deferred-with-rationale` when closure has a clear prerequisite
that's out of phase scope).

## Resolution-quality rules

1. **Impact provenance** — when lens-1 (or any later walk) changes
   a finding's impact, preserve the older rating in parens:
   `HIGH (lens-1; was MEDIUM per lens-3)`.  Don't silently rewrite.
2. **Limitation documentation** — when a closure invokes "out-of-
   scope" or "deferred-with-rationale", the closure record MUST cite
   the prerequisite (staking examples in the examples archive).
   Limitations live in the finding body, not as silent gaps.
3. **LaTeX spec-style separation** (inherited from
   `policy/CONFORMANCE_POLICY.md` §Resolution-quality rule 3):
   `{Project}Spec.tex` is a self-contained mathematical specification
   — NO `F-NN` references, NO `(session N, YYYY-MM-DD)`, NO
   "mirroring the implementation's X" / target file:line, NO
   `Resolved Questions` artifact sections.  Closure history lives in
   `STATUS.md` / `findings/DIVERGENCE_FINDINGS.md` / git, not in the
   spec doc.
4. **Test-name discipline** — differential regression files are
   named `test_{project}_f<NN>_<short_name>.ml` and listed in
   `test/regressions/dune` with a comment block citing the finding
   number, impact, and closure date.

## Settled milestone

An audit phase (lens 3 / lens 1 / lens 2) is **complete** when every
in-scope finding has reached a terminal status (per §Status enum).

The audit overall is **closed** when:

- Every F-NN has terminal status.
- Every lens-2 composition (C-NN) entry has been re-rated under
  lens-1 (impact may stand or change; either is acceptable, but
  un-rated entries block closure).
- The `## Lens 1 — scenario-model summary` section in
  `findings/DIVERGENCE_FINDINGS.md` is up to date with the top-N
  maintenance-window backlog.
- The `## Divergence audit` section in `STATUS.md` reflects the
  current state.

A "fix-before-next-phase" gate may be optionally invoked to require
HIGH-rated open findings to close before the next roadmap phase
starts; this is user discretion, not policy-mandated.

## Pending-bucket triage sweep

Codified 2026-05-08 from a staking gate-closure window.  At audit
close (settled-milestone), the doc-level visibility of findings
benefits from a **summary status table** at the top of `## Findings`
grouping every F-NN by status badge (CLOSED / pending / out-of-scope)
and impact-sorting within each bucket.  The **pending bucket**
(terminal-but-mitigation-pending: triaged with no current G-goal
reach, deferred-with-rationale, or otherwise non-`closed`
non-`out-of-scope`) is a maintenance signal; before moving to the
next roadmap phase, the implementer SHOULD sweep the pending bucket
to convert each entry to one of:

- **close-as-verified** — finding has no spec change required (no-
  action verified during the lens-1 walk).  Status flips to
  `**CLOSED YYYY-MM-DD** (verified-no-action during pending-bucket
  triage sweep; <one-line rationale>)`.  Body status field gains a
  short rationale: "harness-hygiene flag", "migration-only", "latent
  — internal target optimization", "calibrated drift absorbed by
  harness", "subsumed by <existing spec construct>", "defensive
  coding flag against future drift", etc.  Convention parallel to
  `closed (no-finding; verified during task #N walk)` from in-walk
  absorptions.
- **out-of-scope re-classification** — finding has its closure
  path in another layer or audit.  Status flips to
  `**out-of-scope YYYY-MM-DD** (re-classified during pending-bucket
  triage sweep; closure path lives in <layer/audit name>)`.  Impact
  stands but is now read as "would be X if <layer> were in scope".
- **substantive closure** — finding has a real spec change but was
  not selected during the audit's main closure phases.  The atomic
  closure rule applies (per §The atomicity rule); land it now if
  in scope, else mark `deferred-with-rationale` with a clear
  prerequisite + `Remove when:` predicate.
- **architectural verified-no-action** — finding's closure
  requires a fundamentally different spec architecture (e.g.\
  parameters-as-state instead of Section variables).
  Status flips to `**CLOSED YYYY-MM-DD** (verified-no-action;
  architectural decision per <design-stage citation>; <re-open
  predicate>)`.  This is `close-as-verified` with an architectural
  rationale; the `Re-open when:` predicate cites the prerequisite
  for re-opening (typically a generator or harness extension that
  makes the architectural lift test-relevant).

The sweep is documented as ONE STATUS.md entry summarizing the
batch, not 22+ per-finding entries.  The summary table at the top
of `## Findings` rebuilds with the post-sweep counts.

Each pending entry must reach a terminal disposition; `triaged` is
acceptable to retain only when none of the four routes above
applies AND the lens-1 verdict has been reviewed against the
current code.  Bare "still pending, will fix later" is a sweep
discipline failure.

## In-scope cluster pattern *(distinct from out-of-scope clusters)*

Codified 2026-05-08 from a staking cluster (see the examples archive).
Some findings ARE in-scope (they describe gaps in this project's spec)
but their faithful mitigation requires a not-yet-existing **companion
bridge spec** in another formal project's domain.  These findings
share a closure-path dependency, not a root-cause out-of-scope-ness.

**Cluster pattern**: catalogue them in a `## Cluster: depends on
<bridge> bridge` section (NOT `## Out-of-scope cluster`) parallel
to out-of-scope clusters but structurally distinct.  Header MUST
state explicitly that:

- These findings are IN scope (about this project's spec, not
  another project's domain).
- What they share is a closure-path dependency on the bridge spec.
- Cross-references to closed-with-residual entries elsewhere in
  the doc whose closures stand but leave residual gaps in the
  bridge's territory.

**Coverage-taxonomy row**: a one-row pointer in the
`### Out-of-scope by design (potentially in scope for broader analysis)`
table at the top of `findings/DIVERGENCE_FINDINGS.md`, citing the
cluster section and listing the bridge dependency.

**Status field convention**: cluster members MAY use `triaged
(out-of-scope; <bridge> dependency)` OR `deferred-with-rationale
(closure path is <bridge> bridge spec)`.  Both are terminal per
`§Status enum`.  The summary status table at the top of `## Findings`
displays cluster members in the **out-of-scope** bucket alongside
true out-of-scope findings — readers scanning for "what's left to do
at this project's layer" treat cluster routing as out-of-scope at
this layer.

**When the bridge spec lands**: cluster findings re-open via the
`Remove when:` predicate; each finding then gets a normal atomic
closure that threads through the bridge.

## Verification-absorption pattern

When a later walk verifies / closes an earlier walk's pending
verification, **update the original finding's body in-place rather
than creating a new one** (staking examples in the examples archive).

Status-field convention for absorbed entries:
`open (verify X first)` → `closed (verified during task #N walk)`.

## Out-of-scope cluster pattern

When a finding's root cause is in **another** formal project but its
surface is in **this** project's modules, group it in a dedicated
`## Out-of-scope cluster: <other project>` section between Findings
and Lens 2 triage.  Add a one-row pointer in the coverage-taxonomy
"Out-of-scope by design" table.  Impact reads `out-of-scope; would be
X if <other> were in scope` so the count of "open findings" remains a
metric of work this project owns.  See
`memory/feedback_audit_doc_scope.md` for the full convention.

## Lens-2 composition triage format

```
### G<n> — <scenario goals name>

| ID | Path | Composes | Impact | Reach |
|---|---|---|---|---|
| C<n> | <one-line attack name> | F-X, F-Y, ... | <SEV> | <one-line reach note> |
```

- One row per composition path.
- "Reach" column carries the explanation; keeps cells readable.
- Group by scenario goals; add Migration & Cross-domain
  sections at the end.
- Lens-1 ratifies provisional severities at lens-3 close; severities
  change only when lens-1 surfaces new reachability info.

## Pre-closure verdict re-verification (lens-3 staleness rule)

**Before starting closure work on a finding, the implementer MUST
re-verify the lens-3 verdict against current code.**  Codified
2026-05-08 from staking second-pass closures where stale verdicts
surfaced repeatedly (worked examples in the examples archive).

### Why

Lens-3 verdicts are point-in-time observations recorded during the
mechanical walk.  Between the walk and the closure work, other
project phases may land changes that:

- Invert the gap — the bug described in the verdict no longer
  matches current code.
- Supersede the gap — a later phase already closed the finding
  without the audit doc catching up.
- Partially supersede the gap — infrastructure for closure already
  exists; only some of the closure work remains.

Acting on stale verdicts wastes closure effort, ships unnecessary
spec extensions, or worse, ships a closure that doesn't match what
the user actually needs.  The verdict is a HYPOTHESIS, not a
specification — current code is the specification.

### How

Before opening any closure tasks for finding F-NN:

1. **Read the spec file(s) the verdict cites** — confirm the
   claimed pattern (e.g.\ "spec sets X := Y at every tick").
2. **Read the target file(s) the verdict cites** — confirm the
   target's behavior matches the verdict's contrast.
3. **If verdict matches current code**: proceed with planned closure.
4. **If verdict is stale (inverted, superseded, or partial)**:
   - Update the finding's body in `findings/DIVERGENCE_FINDINGS.md`
     to reflect what's actually true today (preserve original verdict
     under a "Lens-3 verdict (historical)" sub-bullet for
     traceability).
   - Re-derive the closure plan based on accurate state.
   - Surface the staleness to the user before committing —
     scope may have changed.

### Triage outcomes

The reconnaissance produces one of three triage outcomes, all valid
under this policy:

- **Atomic closure** — verdict stands; planned closure work proceeds
  per the atomicity rule.
- **Verification-absorption** — verdict is fully superseded; finding
  is closed in-place by citing the earlier phase's work + a
  differential test that pins the property.  Per
  §Verification-absorption pattern.
- **Re-scoped closure** — verdict is partially stale; closure
  proceeds on the residual gap, with the verdict body updated to
  reflect the new scope.

## Post-policy retroactive walk

**Whenever a new rule is added to this policy** (per `policy/CONFORMANCE_POLICY.md
§Policy-accrual discipline`), the implementer MUST re-walk every
closed finding against the new rule and patch any gaps in the same
session.  Codified 2026-05-08 from the experience of this policy's
own birth: the atomicity rule was extracted from five staking
closures, and re-walking those closures against the freshly written
policy surfaced two retroactive gaps (missed simulation-record
slots), both patched in the same session that landed the policy
(details in the examples archive).

The retroactive walk is non-negotiable: a new policy rule that does
not surface ZERO gaps means either (a) the rule is too narrow to be
useful, or (b) the existing closures were lucky to not violate it
yet.  Either way, the walk is what makes the new rule load-bearing.

Output shape: when running the retroactive walk, name each closure
checked and either confirm compliance or land the fix in-line.  Do
not defer to a later session.

## Cross-references

- `policy/CONFORMANCE_POLICY.md` — sibling policy on spec ↔ target
  conformance.  Both apply concurrently.
- `findings/DIVERGENCE_FINDINGS.md` — the audit document this policy
  governs.
- `STATUS.md` §Divergence audit — per-closure record.
- `memory/feedback_divergence_audit_workflow.md` — three-lens
  framework + finding template (the patterns this policy formalises).
- `memory/feedback_divergence_finding_closure_atomicity.md` — the
  atomicity rule's source memory.
- `memory/feedback_audit_doc_scope.md` — per-project scope rule +
  out-of-scope cluster pattern.
- `memory/feedback_next_step_discipline.md` — lens-1 mandated next
  step before audit phase closes.
- `memory/feedback_latex_spec_style.md` — LaTeX spec-style
  separation rule.
- `memory/feedback_formal_project_docs.md` — STATUS / FILES /
  METHODOLOGY / JOURNAL pattern; divergence audit slots alongside.
- `examples/tezos/divergence_examples.md` — the staking worked
  examples this policy's rules were extracted from.

## History

- 2026-05-08 — Policy created in the staking project.  Atomicity rule
  (§The atomicity rule) added as the lead requirement after five
  staking lens-1 closures revealed the 2-phase pattern as an
  anti-pattern.  Three-lens framework, finding template, status enum,
  resolution-quality rules, settled-milestone,
  verification-absorption, out-of-scope cluster, lens-2 triage format
  consolidated from the divergence-audit-workflow memory.
- 2026-06-11 — Generic core extracted (apparatus v1.1); staking
  worked examples → `examples/tezos/divergence_examples.md`; the
  fulfilled "Productisation TODO" section retired (the apparatus IS
  the productisation: lens phases are sub-commands, the findings
  template ships in `project-template/findings/`, and this file is
  the generic policy it called for).
