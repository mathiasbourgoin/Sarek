---
name: roster-spec-infer
description: Reverse-engineer what existing code actually does into a structured, evidence-tiered inferred spec artifact (specs/<slug>-inferred.md). Mines tests, code, documentation, and git history. Labels every claim [E1]/[E2]/[E3] by evidence strength. Produces GWT scenarios, FR-NNN requirements, and surfaces doc-drift conflicts.
version: 1.0.0
tunables:
  include_git_history: true
  conflict_policy: surface
  gap_threshold: 0
artifacts:
  writes:
    - specs/<slug>-inferred.md
allowed_tools: [Read, Bash, Agent]
---

# Roster Spec-Infer

Reverse-engineer an evidence-tiered spec from an existing codebase. This is a
read-only analysis skill — it never modifies the target project. Every claim is
labeled by evidence strength: `[E1]` (test-proven), `[E2]` (code-inferred),
`[E3]` (doc-only). Conflicts between code and documentation are surfaced, never
silently resolved.

## Input Contract

**Required:** `<target>` — a project root directory path (e.g. `.` or `/path/to/project`).

- If a **file path** is given: halt with "Target must be a directory (project root). Got: `<path>`."
- **Slug derivation:** lowercase the target directory's basename, replace spaces with `-`,
  strip all non-alphanumeric/hyphen characters. Example: `My Project!` → `my-project`.
- **Collision check:** if `specs/<slug>-inferred.md` already exists, halt with:
  "Inferred spec already exists for `<slug>`. Overwrite? [y/N]" — stop if not confirmed.

**Tunables:**

| Tunable | Default | Effect |
|---|---|---|
| `include_git_history` | `true` | Run Sub-agent 4 (Git History Miner) |
| `conflict_policy` | `surface` | Only valid value in v1.0.0: always surface conflicts, never auto-resolve. Reserved for future extension. |
| `gap_threshold` | `0` | Register every uncovered public symbol as a GAP |

## Steps

### Sub-agent 1: Test Miner

Spawn a sub-agent (fresh context) with this prompt:

```
You are a Test Miner. Analyze the test suite of the target project and extract
behavioral assertions as evidence-tier-1 (E1) claims in Given/When/Then format.

Step 1 — Detect test framework (in this priority order):
- package.json `scripts.test` field
- pytest.ini / setup.cfg / pyproject.toml [tool.pytest]
- go.mod (go test ./...)
- pom.xml / build.gradle (mvn test / gradle test)
If no framework detected: emit "No test suite detected — all surfaces are E2/E3 or GAP."
and produce zero E1 claims. Stop.

Step 2 — Run the canonical test command. If exit code is non-zero: halt with:
"Test suite failed — E1 claims cannot be emitted. Fix test failures before
running roster-spec-infer."

Step 3 — Parse passing test files. For each behavioral assertion, produce one
E1 entry in Given/When/Then format:
  - [E1] Given <setup/precondition>, When <action or call>, Then <assertion/outcome>
    (evidence: <test-file>:<line>)

Use the actual test setup code as "Given", the call under test as "When", and
the assertion as "Then". If a test has multiple assertions, emit one E1 entry
per distinct behavioral claim, not per assertion line.

Cap at top-50 most-covered files if context limits are reached; note:
"Analysis limited to top N files due to context constraints."

Target directory: <target>
```

Collect E1 claims in GWT format. If test suite is absent or fails, proceed with
empty E1 set — note this prominently at the top of the output artifact.

---

### Sub-agent 2: Code-Only Analyzer

Spawn a sub-agent (fresh context) with this prompt:

```
You are a Code-Only Analyzer producing E2 claims.

CRITICAL EXCLUSION: Your context contains ONLY implementation source files.
The following are EXCLUDED and must NOT be referenced:
- ALL .md files and README files
- Docstring content (Python """ """, JSDoc /** */, Ruby =begin/=end)
- Any comment explaining what a function does for external readers
  (internal "why" comments about implementation decisions may be included)
- Any file whose primary purpose is documentation

For each public/exported symbol or module boundary, infer behavioral claims
from the code logic alone. Produce:
  - [E2] <claim text> (evidence: <file>:<line>)

Cover every public/exported symbol even if it has no E1 coverage — these
uncovered symbols will become GAP candidates.

Target directory: <target>
```

Collect E2 claims. Note all public/exported symbols identified — used for gap detection.

---

### Sub-agent 3: Code+Docs Analyzer

Spawn a sub-agent (fresh context) with this prompt:

```
You are a Code+Docs Analyzer producing an independent claim set for conflict detection.
(TRACE — Two-Reader Agreement with Corroborated Evidence — runs two independent analyzers
to surface claim divergence caused by documentation drift.)

Read all source files PLUS all documentation: .md files, READMEs, docstrings, JSDoc.

For each public/exported symbol, produce claims:
- [E2] <claim text> (evidence: <file>:<line>)   ← when claim comes from code
- [E3] <claim text> (evidence: <file>:<line>) [E3: lowest confidence — verify against code]
  ← when claim comes solely from documentation with no code corroboration

If no documentation files found: note "No documentation found — conflict detection
not applicable" and produce output identical to the Code-Only Analyzer.

Target directory: <target>
```

Collect this independent claim set. Do NOT merge it with Sub-agent 2's output yet —
the orchestrator will diff them.

---

### Sub-agent 4: Git History Miner *(skipped if `include_git_history: false`)*

Spawn a sub-agent (fresh context) with this prompt:

```
You are a Git History Miner. Annotate behavioral claims with design-decision
context from commit history.

1. Collect the list of key implementation files identified by Sub-agents 2 and 3
   (files containing the most public/exported symbols). Pass this concrete list to
   Sub-agent 4.
2. Run: `git log --follow --oneline -- <collected file list>`
3. Run: `git log --oneline -50`

If not a git repo or no history: emit "Git history unavailable — commit
annotations omitted." and return empty.
If shallow clone: note "Shallow clone detected — history may be incomplete."

For each commit that explains a behavioral decision (not just refactors or
style fixes), produce:
  - commit:<sha7> <implementation area> — <behavioral decision explained>

Target directory: <target>
```

Collect commit annotations keyed by implementation area.

---

### Orchestrator Synthesis

After all sub-agents complete:

**1. Conflict detection (TRACE — Two-Reader Agreement with Corroborated Evidence)**

Diff Sub-agent 2 (code-only) vs Sub-agent 3 (code+docs) claim sets:
- For each claim where the two describe the same symbol differently: create a CONFLICT entry:
  ```
  CONFLICT-N: <symbol or behavior area>
    Code-Only: <Sub-agent 2 claim text> (evidence: <file>:<line>)
    Code+Docs: <Sub-agent 3 claim text> (evidence: <file>:<line>)
    Note: [CONFLICT: doc-drift suspected] — human adjudication required.
  ```
- Remove conflicting claims from `## Claims` — they belong ONLY in `## Conflicts`.
- For non-conflicting claims where Sub-agent 3 adds documentation corroboration to a
  Sub-agent 2 claim: append the doc citation to the existing E2 claim's evidence field:
  `(doc: <file>:<line>)`. No label change — the claim stays [E2].
- If Sub-agent 3 found no documentation ("No documentation found"): conflict detection
  is not applicable; note "No documentation found — conflict detection not applicable"
  and proceed with Sub-agent 2's claims as the sole source.

**2. Gap registration**

For each public/exported symbol identified by Sub-agents 2/3 with zero E1 claims:
```
GAP-N: <symbol or module name> — no E1 evidence; behavior unverified by tests.
```
If `gap_threshold > 0`: only register gaps for modules with ≥ threshold uncovered symbols.
If no exported symbols detected: note "No public API surface detected."

**3. Git annotation merge**

For each E1/E2 claim corresponding to an area with a commit annotation, append:
`(commit:<sha7>)` to its evidence field.

**4. Requirements Formalizer**

After conflict detection and gap registration, spawn a sub-agent:

```
You are a Requirements Formalizer. You receive E1/E2/E3 claims and challenge resolutions
from a reverse-engineered codebase analysis.

Your job: produce FR-NNN MUST/MUST NOT statements — one normative requirement per
distinct behavioral obligation, grouped by module or feature area.

Rules:
- Derive FRs ONLY from E1 and E2 claims (not E3 — E3 is unverified documentation)
- Use MUST for observed/tested behaviors (prefer E1 sources)
- Use SHOULD for inferred behaviors with only E2 support
- Do NOT include FRs for CONFLICT entries — conflicts must be resolved by humans first
- Each FR must reference its evidence: [E1:<file>:<line>] or [E2:<file>:<line>]
- Mark clearly: FRs from E1 are higher confidence than FRs from E2 alone

Format:
#### <Module or Feature Area>
- **FR-001** [E1]: System MUST <normative statement> [E1:<test-file>:<line>]
- **FR-002** [E2]: System SHOULD <inferred behavior> [E2:<source-file>:<line>]
- **FR-003** [E1]: <actor> MUST NOT <prohibited behavior> when <condition> [E1:<test-file>:<line>]

E1/E2 claims:
<all non-conflict claims from orchestrator>
```

**5. Write output artifact** to `specs/<slug>-inferred.md`.

## Output Contract

The skill writes one file: `specs/<slug>-inferred.md`.

**Frontmatter:**
```yaml
---
name: roster-spec-infer
type: inferred-spec
status: draft
feature: <slug>
inferred_by: roster-spec-infer
date: <ISO-8601>
version: 1.0.0
---
```

**Sections (in order):**

1. `## Claims` — all non-conflicting E1/E2/E3 claims in GWT format (E1) or prose (E2/E3),
   grouped by module/feature area
2. `## Functional Requirements` — FR-NNN MUST/SHOULD statements from the Requirements
   Formalizer, grouped by module (E1-sourced FRs listed before E2-sourced)
3. `## Gaps` — all GAP-N entries (mandatory; "No gaps detected." if none)
4. `## Conflicts` — all CONFLICT-N entries (mandatory; "No conflicts detected." if none)

**Completion messages (print at end of run):**

1. `"⚠️  This artifact is inferred, not authoritative. Do not use as a ground truth for new implementations."`
2. `"Run /spec-compliance-auditor specs/<slug>-inferred.md to detect behavioral drift against the current implementation."`

### What Next

- `/spec-compliance-auditor specs/<slug>-inferred.md` — audit implementation against this inferred spec
- `/roster-spec` — produce a forward spec that incorporates the inferred findings as a starting point

## Rules

1. Evidence labels: `[E1]`, `[E2]`, `[E3]` only — never "Tier 1/2/3" (reserved for Ralph Loop quality gates).
2. **E1** requires: (a) a test assertion found in a test file AND (b) the test suite exits 0.
3. **E2** is best-effort static inference. Not equivalent to E1.
4. **E3** annotation `[E3: lowest confidence — verify against code]` is mandatory on every E3 claim.
5. CONFLICT entries appear ONLY in `## Conflicts` — never in `## Claims`.
6. Absence of conflicts does not certify correctness. Both analyzers may agree on a wrong claim if the bug exists in both code and docs simultaneously (TRACE blind spot — not a substitute for human review).
7. Gap registration is mandatory — every uncovered public surface is a GAP-N entry. Silence never implies completeness.
8. FRs derived from E3-only sources are prohibited — E3 claims are documentation assertions, not behavioral evidence.
9. Output frontmatter must contain `type: inferred-spec` and filename must end in `-inferred.md`.
10. This skill is read-only with respect to the target codebase. Only `specs/<slug>-inferred.md` is written.
11. Cross-module behavioral invariants requiring simultaneous knowledge of multiple modules may not appear in E2 claims (V1 limitation).
