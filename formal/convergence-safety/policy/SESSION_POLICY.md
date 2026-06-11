# Session Policy — AI Agent Context Management

**Standing policy** (2026-05-05; generic core extracted 2026-06-11,
apparatus v1.1 — locked semantics unchanged). Sibling to
`policy/CONFORMANCE_POLICY.md` and `policy/BENCHMARK_POLICY.md`.  This
file is the single authority on how AI agents manage context, memory,
and session boundaries when working on the formal project.

## Why this exists

Long agent sessions accumulate context that is expensive to reconstruct.  Some
of it is structured (commits, files, tests) and recoverable from the repo.
Some of it is judgment, observation, and trajectory data that lives only in
the conversation and evaporates when the session closes.

The policy distinguishes:
- **Memory hygiene rules** — enforceable, no judgment.  Always followed.
- **Session-end checklist** — enforceable, runs before closing a session.
- **Coupled-vs-decoupled heuristic** — recommendation that backs the agent's
  suggestion of "continue here" vs "open a fresh session".

## Memory file layout

The auto-memory directory is `{auto_memory_root}` (the host user's
per-project Claude memory directory, e.g.
`~/.claude/projects/<project-slug>/memory/`).  It contains one index
file plus one file per memory entry.

| File | Role | Format |
|---|---|---|
| `MEMORY.md` | Index | One pointer line per entry, ≤200 chars |
| `project_<topic>_<context>_findings.md` | Topic memory for a finding closure | Self-contained; frontmatter `name/description/type` |
| `project_<topic>_<task>_onboarding.md` | Hand-off pointer for a future session | Self-contained; HEAD ref + diagnostic plan + file:line pointers |
| `feedback_<topic>.md` | User-given guidance to repeat or avoid | Rule + **Why:** + **How to apply:** |
| `user_<aspect>.md` | User profile / preferences | Self-contained |
| `reference_<system>.md` | Pointer to external resource | Where + when-to-consult |

`MEMORY.md` is an INDEX.  It never carries body content.  Body content lives
only in topic files.

## Memory hygiene rules

These rules are **always followed** when working in the formal project.

### 1. Topic file per finding closure

After every atomic-finding commit (per `policy/DIVERGENCE_POLICY.md`
§The atomicity rule), write or update a topic memory file documenting:
- Closure commit SHA
- Diagnosis (root cause, not just symptoms)
- The fix (1-3 lines describing the change)
- Empirical residual (numerical, with units)
- Any lessons learned that don't belong in code comments

Filename: `project_{project}_session<N>_findings.md` for a session-spanning
record, or `project_{project}_finding_<id>.md` for a single-finding record.
Choose by whether the finding stands alone or is part of a multi-finding
session arc.

### 2. Index pointer in MEMORY.md

Same trigger.  Add a single line under the project's section:

```markdown
- See `project_{project}_session<N>_findings.md` for **<headline>** (<date>, COMMITTED `<SHA>`): <one-line summary>.
```

Index lines are ≤200 chars.  If the summary won't fit, the topic file is the
right place for detail.

### 3. HEAD reference update

After every commit on the working branch, update the `Current branch:` line at
the top of `MEMORY.md` with the new HEAD SHA and a one-line lineage.

### 4. Absolute dates only

Always write `2026-05-05`, never "today" / "yesterday" / "last week".  Memory
records outlive the session that wrote them.

### 5. Self-contained topic files

Topic files do not reference the conversation that produced them.  They cite
file paths, function names, commit SHAs, env vars, empirical numbers — never
"as we discussed" / "per my earlier message".

### 6. No restatement of derivable facts

Per the system prompt's "What NOT to save in memory":
- Don't restate code structure / file paths / commit history derivable from `git log`.
- Don't store debugging recipes — the fix is in the code.
- Don't store ephemeral task state — that's what tasks are for.

Save what is **surprising, non-obvious, or judgment-laden**.

### 7. Verify line numbers

Line:file pointers in memory drift as the codebase evolves.  When citing one,
either:
- Verify it still resolves before writing the entry, OR
- Cite by symbol (function name) and let future sessions grep.

A function name is a stable handle; a line number is a handle that decays.

## Session-end checklist

Before closing a working session, verify:

- [ ] All commits on the working branch are atomic per `policy/DIVERGENCE_POLICY.md` §The atomicity rule.
- [ ] The host formatter (`{fmt_cmd}`) is clean on touched files.
- [ ] HEAD reference in `MEMORY.md` reflects the latest commit on the working branch.
- [ ] If at least one finding closure landed: a topic memory file exists or was updated, with a corresponding index pointer.
- [ ] If the next investigation is non-trivial (see heuristic below): an onboarding pointer file is written.
- [ ] No diagnostic instrumentation is left in committed code (env-var-gated traces are OK and stay; one-shot dumps inserted for diagnosis must be removed before commit).

## Coupled-vs-decoupled heuristic

When the user asks "continue here or open a fresh session?", recommend based
on whether the next task is **coupled** to the current investigation.

### Continue (recommend staying)

Recommend staying when **any** of these holds:
- Same scenario or fixture is under investigation.
- The next fix is a direct continuation of the current diagnostic chain
  (e.g., consecutive finding closures in one lineage).
- Cached context (function structure, recently-read files, in-flight numerical state) is load-bearing for the next step.
- The task is small (~30 minutes of focused work).

### Fresh session (recommend opening new)

Recommend a fresh session when **all** of these hold:
- Different scenario / different code paths than the just-finished work.
- Different hypothesis frame (e.g., a rounding hypothesis vs a
  scheduling hypothesis vs an accounting hypothesis).
- The next investigation is non-trivial (multiple hours, multiple files).

When in doubt, **state the trade-off and ask the user**.  Context cost is
visible to them; judgment is theirs to make.

### Pre-pay before closing

When recommending fresh session, write an **onboarding pointer file**
(`project_{project}_finding_<id>_onboarding.md`) before closing.  Include:
- HEAD SHA + finding lineage closed this session
- Empirical fact to start from (last observed gap, with numbers)
- Reproduction recipe (exact `dune exec` command + env vars)
- Diagnostic env var inventory (which vars exist, what they do, where defined)
- File:symbol pointers (not file:line — see hygiene rule 7)
- Hypothesis ladder ranked by likelihood, with diagnostic plan
- Cross-reference to relevant existing topic memory files

This is ~5 minutes of writing now and saves ~15-20 minutes of re-discovery in
the next session.

## What this policy does NOT cover

- **Tool selection** — that's covered by the system prompt's tool-use guidance.
- **Code-style / formatting** — the host formatter (`{fmt_cmd}`) is authoritative.
- **Spec-vs-target classification of findings** — see `policy/CONFORMANCE_POLICY.md`'s `(a)/(a')/(b)/(c)` taxonomy.
- **What benchmarks count** — see `policy/BENCHMARK_POLICY.md`.

## Cross-references

- `policy/CONFORMANCE_POLICY.md` — what counts as a conformance test, finding classification.
- `policy/DIVERGENCE_POLICY.md` — the atomic-closure norm the checklist enforces.
- `policy/BENCHMARK_POLICY.md` — bakeoff structure, ledger requirements.
- `{repo_agent_doc}` (host repo top level) — repository-wide agent guidance.
- `examples/tezos/session_examples.md` — the staking memory-file examples
  the v1.0 policy cited.

## Revisions

- 2026-05-05 (staking session 49): initial draft, born from the session that
  closed three findings and handed off the next.
- 2026-06-11: generic core extracted (apparatus v1.1); staking examples →
  `examples/tezos/session_examples.md`; the atomicity-rule cross-reference
  corrected to `policy/DIVERGENCE_POLICY.md` (the v1.0 text cited a
  nonexistent `CONFORMANCE_POLICY.md §3b`).
