---
name: roster-run
description: Pipeline entry point — detects context and routes to the right skill.
version: 1.5.0
---

# Roster Run

You are the entry point of the roster pipeline. Your only job is to detect context and route to the appropriate skill — not to do the work yourself.

## Three modes — pick before anything else

**Read the task first. Classify it. Then route.**

| Mode | When | Pipeline |
|---|---|---|
| **Express** | No spec/KB impact — typo, rename, formatting, config tweak, dependency bump, doc fix, pure refactor with no behaviour change | implement → review → ship |
| **Fast** | Quick task with potential spec/KB impact — bug fix, small behaviour change, adding a missing case, performance fix | implement → review → qa → (update KB/specs/friction log) → ship |
| **Full** | New capability, API change, design decisions, multi-file refactor with trade-offs, anything the user asks to spec first | question → research → intake → spec → plan → implement → review → qa → ship |

**When in doubt between Express and Fast, pick Fast.** When in doubt between Fast and Full, ask one question: "Does this require deciding *what* to build, or just *how*?" — if only *how*, stay Fast.

> Express and Fast are not shortcuts on quality — review is always mandatory. What changes is the upfront discovery and downstream documentation overhead.

### Express signals (all must apply)

- No new behaviour — same inputs produce same outputs after the change
- No spec, KB, or friction log update needed
- Change is self-evident from the task description alone

### Fast signals (any one is enough)

- Fix to existing behaviour (bug, edge case, missing guard)
- Small addition that doesn't change the overall design
- User says "quickly", "fast", "small", "just fix"
- Task ≤ 20 words and unambiguous but has some spec/KB impact

### Full signals (any one is enough)

- New capability that doesn't exist yet
- API or interface change affecting callers
- Multiple design trade-offs to resolve
- User says "feature", "spec", "design", "plan", "implement from scratch", or asks a question about *what* to build

## Hook Execution

Before routing to a skill, check for skill hooks. Hooks are executed by you (the LLM agent) — not by a separate process.

### Non-Reentrance Guard

**Before executing any hook:** check whether `HOOK_RUNNING` is set in your current context. If it is, skip hook execution silently for all nested skill invocations — do not error. This is a prose convention, not a process mechanism.

**When executing a hook:** set `HOOK_RUNNING: true` in your context for the duration of hook execution, then clear it after.

### Discovery (before routing)

1. Determine the `name:` frontmatter field of the target skill file in `.harness/skills/<skill-file>.md` — this is the lookup key, **not** the routing slug.
2. Check if `.harness/hooks/skills/<name>/pre.md` exists.
3. If present **and** `HOOK_RUNNING` is not set in current context: execute the pre-hook (see execution instructions below).

### Pre-Hook Execution

1. Read the hook `.md` file (use `.inlined.md` variant if present, fall back to original).
2. Extract the `steps:` fenced YAML block.
3. Execute each step in order by type using the step-type dispatch table below.
4. If any step fails and `on_error:` (step-level, or hook-level default `stop` for pre-hooks) is `stop`:
   - Print: `Hook <hook-name> aborted at step <N> (on_error: stop) — skill dispatch cancelled. Hook output: <stdout>`
   - Do **not** dispatch the skill.
5. If `on_error: warn` — log the failure and continue.
6. If `on_error: skip` or `skip-step` — skip the current step, continue.
7. If `on_error: retry:N` — retry up to N times before applying the next level default.
8. If `on_error: ignore` — silently continue.

### Post-Hook Execution

1. After the skill completes (regardless of skill outcome), check for `.harness/hooks/skills/<name>/post.md`.
2. If present and `HOOK_RUNNING` is not set: execute similarly to pre-hook.
3. Pass skill outcome as implicit context (available in `prompt:` steps).
4. Default `on_error:` for post-hooks is `warn` (log and continue — do not retroactively affect skill outcome).

### Step-Type Dispatch

The hook executor (`scripts/run-hook.ts`, CLI: `node dist/scripts/run-hook.js <pre|post> <skill>`) enforces real execution for shell steps. Call it before routing for pre-hooks and after the skill completes for post-hooks.

```bash
node dist/scripts/run-hook.js pre <skill-name>
# exit 0=pass  1=abort (skip dispatch)  2=warn  3=pending_llm_steps  4=skip (no hook)
```

For steps the runner returns in `pending_llm_steps` (prompt:, loop:, parallel:), execute them as LLM-interpreted steps after reading the JSON output.

| Step operator | Executed by | Behaviour |
|---|---|---|
| `run: <cmd>` | **Runner (real shell)** | Enforced exit code, real timeout via AbortController, retry loop |
| `test: <cmd>` | **Runner** | Real exit code → on_true / on_false branch |
| `timeout: <ms>` | **Runner** | Updates shell timeout for all subsequent `run:` steps |
| `retry: N` + `backoff:` | **Runner** | Real retry loop with setTimeout backoff |
| `log: <text>` | **Runner** | process.stderr.write — always fires |
| `label: <name>` | **Runner** | Jump target (index-based) |
| `goto: <label>` | **Runner** (intra-hook) / **LLM** (pipeline) | Intra-hook: index jump. Pipeline target: returned in pending_llm_steps |
| `on_error: stop/warn/skip/ignore` | **Runner** | Enforced by exit code logic |
| `prompt:` + `agent:` | **LLM** (pending_llm_steps) | Returned by runner, executed by agent |
| `loop:` | **LLM** (pending_llm_steps) | Returned by runner, executed by agent |
| `parallel:` | **LLM** (pending_llm_steps) | Sequential in v1 |
| `include:` | Build-time (sync-harness.sh) | Already inlined as `.inlined.md` variant |
| `output:` | Metadata | Noted, not enforced |

## Routing

**Step 1 — classify the task (Express / Fast / Full).** Do this before checking briefs/.

If **Express** mode: announce and route directly through **implement → review → ship**.

If **Fast** mode: announce and route through **implement → review → qa → ship** in sequence. After QA, update KB/specs and friction log if impacted.

If Full mode: check briefs/ state and use the routing table below.

### Full-mode routing table

| Detected signal | Route to |
|---|---|
| No brief, new feature, vague or multi-file task | `/roster-question` (then research → intake) |
| `briefs/<task>-intake.md` VALIDATED + `**Type:**` is feature/api-change + `briefs/<task>-spec.md` absent | `/roster-spec` |
| `briefs/<task>-spec.md` present with status `BOUNCED` | `/roster-intake` — enrich the brief to resolve the bounce reason, then re-run `/roster-spec` |
| `briefs/<task>-intake.md` exists and is validated | `/roster-plan` |
| `briefs/<task>-plan.md` exists and is validated | `/roster-implement` |
| Implementation complete, branch ready | `/roster-review` |
| `briefs/<task>-review.json` with GO status | `/roster-qa` |
| `briefs/<task>-review.json` with NO-GO + `no_go_reason.type == "spec-ac-failure"` | `/roster-spec` — spec ACs were not met; revise the spec |
| `briefs/<task>-review.json` with NO-GO (any other reason) | `/roster-implement` — pass review.json as context |
| `briefs/<task>-qa.md` with GO status | `/roster-ship` |
| Complex bug with unclear root cause, no obvious fix | `/roster-investigate` |
| New project or existing project without harness | `/roster-init` |
| Periodic analysis, friction patterns | `/roster-skill-health` |
| No signal matches | Stop — ask the user: "What are we doing?" before routing |

### Detection

1. Check for the existence of `briefs/` artifacts with explicit bash commands:
   ```bash
   ls briefs/ 2>/dev/null || echo "briefs/ absent"
   # Then for the current task:
   [ -f briefs/<task>-intake.md ] && echo "intake: present" || echo "intake: absent"
   [ -f briefs/<task>-spec.md ]   && echo "spec: present"   || echo "spec: absent"
   grep '\*\*Type:\*\*' briefs/<task>-intake.md | head -1
   [ -f briefs/<task>-plan.md ]   && echo "plan: present"   || echo "plan: absent"
   [ -f briefs/<task>-review.json ] && echo "review: present" || echo "review: absent"
   # If review.json is present, read its status and no_go_reason:
   [ -f briefs/<task>-review.json ] && jq -r '"\(.status) \(.no_go_reason.type // "none")"' briefs/<task>-review.json 2>/dev/null
   [ -f briefs/<task>-qa.md ]     && echo "qa: present"     || echo "qa: absent"
   ```
2. Check the status of existing artifacts (GO / NO-GO / absent) — read the first status line of each present file.
3. If `briefs/` is absent or empty and $ARGUMENTS is empty or ambiguous, ask **one single question**:
   > "What are we doing?" (do not propose a list, let the user describe)

### Announce

Before routing, announce in one line:
> "→ [FAST|FULL] mode: <route> because <reason in 5 words max>"

## When to Go Back

| Condition | Action |
|---|---|
| No route matches the current project state | Stop — ask the user to describe the situation before routing |
| Routing would skip a mandatory phase | Route to the earliest upstream phase instead |

## What Next

After routing, the destination skill announces its own **What Next** upon completion — follow that chain.

> 💡 Run `/roster-skill-health` periodically to surface friction patterns and improve the pipeline.

## Rules

- Never do the work of another skill — route only
- Never route to multiple skills in parallel from here
- If no route matches, ask the user before inventing one
