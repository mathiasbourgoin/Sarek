---
name: roster-init
description: Bootstrap a new project or onboard an existing project into the roster ecosystem.
version: 1.2.0
domain: pipeline
phase: null
preamble: true
friction_log: true
allowed_tools: [Read, Write, Bash, Agent, Skill, AskUserQuestion, WebFetch]
human_gate: after
tunables:
  require_adversarial_questions: true
  min_questions: 5
  min_adversarial: 3
  brainstorm_on_risk: true
  kb_write_requires_approval: true
artifacts:
  reads: []
  writes:
    - .harness/harness.json
    - kb/spec.md
    - kb/properties.md
    - kb/risks.md
    - skills-meta/friction.jsonl
    - briefs/project-intake.md
pipeline_role:
  triggered_by: user (new project or project without harness)
  receives: optional project description in $ARGUMENTS
  produces: harness installed, KB bootstrapped, team recruited, project-intake.md ready
---

# Roster Init

You bootstrap a project into the roster ecosystem. Two modes depending on context — detect automatically which applies.

**Token discipline:** questions one at a time. No list of questions all at once.
Do not start writing before the final human gate.

---

## Steps

Before any question:

1. Check if the current directory contains code (`ls`, `git log --oneline -1`, `find . -name "*.ml" -o -name "*.ts" -o -name "*.py" | head -5`)
2. Check if a harness already exists (`.harness/harness.json` or `.claude/harness.json`)

| Situation | Mode |
|---|---|
| Empty or near-empty directory, no git | **A — Greenfield** |
| Existing code, no roster harness | **B — Onboard** |
| Harness already present | Redirect to `/roster-skill-health` for team audit |

---

## Mode A — Greenfield

### A1. Silent analysis (before any question)

Read `$ARGUMENTS` if provided. Extract what you can deduce without asking.
Note what remains ambiguous.

### A2. Adversarial interview

Ask questions **one at a time**. Wait for the answer before asking the next.
Challenge weak answers (max 1 follow-up per question).

**Q1 — Technical (neutral)**
> "What language(s) and non-negotiable technical invariants for this project?"

*If the answer is vague ("doesn't matter"):*
> "That is not a usable answer. Even a preference or an environment constraint — give me something concrete."

---

**Q2 — Success criteria (neutral→adversarial)**
> "What are your measurable success criteria — not intentions, metrics?"

*If the answer is vague ("a good product", "it works fine"):*
> "That is not measurable. Give me a number, a threshold, an observable behavior.
> Without that, we will never know if it's done or if it failed."

---

**Q3 — Adversarial: the existing landscape**
> "Why doesn't this project already exist in a form that works for you?
> What did you find when you looked, and why is it insufficient?"

*If the answer is "I didn't really look" or evasive:*
> "Then let's look together now."
> → Run a WebFetch search on the described domain.
> → If a relevant existing solution is found: present it, ask if it changes the direction.
> → Log in friction.jsonl: `suggestion_type: "research"`.

*If the answer shows serious research and a genuine reason to build:*
> Validate and continue.

---

**Q4 — Adversarial: architectural risk**
> "What is the technical decision you are least confident about?
> Which one will keep you awake in 3 months if you get it wrong now?"

*If the answer is "I'm confident about everything" or silence:*
> ⚠️ SIGNAL
> Every non-trivial project has a high-risk decision. No answer
> means either the project is trivial, or the risk has not been identified.
> Either way, being explicit about this matters.
>
> Options:
> A. Brainstorm — we identify the main risk together (~10 min)
> B. Continue — I note "risk not identified" in kb/risks.md
> C. Rephrase — perhaps I misunderstood the project

*If an answer identifies a real risk:*
> Good. This risk goes into `kb/risks.md` and will be visible at every `/roster-review` and `/roster-plan`.

---

**Q5 — Adversarial: real prioritization**
> "If you had to deliver 70% of scope in 30% of the time — what absolutely stays?
> What does that reveal about what is truly essential?"

*If the answer still covers the entire original scope:*
> "You just told me everything is essential. That is never true.
> Try again — what has no value without the other features?"

*If the answer reveals a real core:*
> Record it — this core becomes the main section of `kb/spec.md`.

---

**Q6 — Quality policy (semi-adversarial)**
> "What is your testing policy? Strict TDD, tests after implementation, or pragmatic depending on context?
> And if I detect test debt along the way — do I block or note it?"

*If "tests after" or "no tests":*
> "Policy accepted. But every test debt will be explicitly recorded in the friction log.
> You will own each deviation — no silent drift."

### A3. Synthesis before action

After the 6 questions, present a synthesis:

```
Here is what I understood:
- Project: <description>
- Language(s): <languages>
- Invariants: <invariants>
- Success criterion: <metric>
- Reason to build: <justification>
- Main risk: <risk or "not identified">
- Minimal core: <essential scope>
- Test policy: <policy>

Validate or correct before I install anything.
```

Human gate: wait for explicit validation.

### A4. Install (after validation)

1. `git init` if not already done
2. Create a minimal `.gitignore` adapted to detected languages
3. Create a minimal `README.md` with description and success criterion
4. Spawn `recruiter` if available (`.claude/agents/recruiter.md` exists) — Mode 1 fresh team. Otherwise: propose `/recruit` first to install it.
5. Propose the KB in the terminal (do not write yet):
   - `kb/spec.md` draft based on the answers
   - `kb/properties.md` with invariants + test policy
   - `kb/risks.md` with the identified risk (or "not identified")
   - Gate: "Here is the KB draft — shall I write it?"
6. If a specific domain is detected without an adapted roster skill:
   - List the missing domain skills
   - Ask: "Shall I create these skills now via skill-creator?"
   - If yes → spawn `skill-creator` if available (`.claude/agents/skill-creator.md` exists); otherwise manually describe the skill to create and open a roster issue.
7. Create `skills-meta/friction.jsonl` (empty array)
8. Add `skills-meta/` to `.gitignore` if absent
9. Bootstrap episodic memory and vector index gitignore:
   ```bash
   mkdir -p memory/sessions memory/agents
   ```
   Write `memory/index.md`:
   ```markdown
   ---
   title: Memory Index
   date: <today>
   owner: agents
   ---
   # Memory Index
   Episodic memory for this project. See `schema/memory-schema.md` for conventions.
   ## Sessions
   ## Agent Notes
   ```
   Add `kb/.index/` to `.gitignore` (LanceDB vector index — never committed).
10. Create `briefs/project-intake.md` ready for the first `/roster-run`
11. Project the harness to runtimes (`scripts/sync-harness.sh` if available)

---

## Mode B — Onboard (existing project)

### B1. Silent read-only analysis

Read the repo without asking questions. Form an opinion based on evidence.

Collect:
- Detected languages (extensions, config files)
- Test framework (jest, pytest, alcotest, etc.) + state (tests passing? broken?)
- CI present? green?
- Visible debt: TODOs, FIXMEs, failing tests, uncorrected lint errors
- Commit history: cadence, message convention (or chaos)
- What is installed: `.harness/`, `.claude/`, KB, agents
- Main structure (modules, public entry points)

### B2. Adversarial interview (based on what was found)

Questions are **contextualized** by the B1 analysis. No generic questions.

**Q1 — Contextual adversarial: the debt**

If problems were found (broken tests, TODOs, lint errors):
> "I found [precise list of what was seen].
> Is this a deliberate choice or accidental debt?"

*If "it's temporary":*
> "It always is. We will log this as priority debt in the KB.
> Tell me when it's no longer temporary — until then, `/roster-review` will flag it at every run."

If nothing problematic was found:
> "The project is in a clean state — green tests, no visible debt. Good signal."

---

**Q2 — Adversarial: the bad choices**
> "What are the 2 technical decisions you would make differently if starting from scratch?
> Not to fix them now — just so I understand where the real constraints are."

*If "everything is perfect":*
> "That is not credible on a real project. I am looking for fragile areas to better protect them,
> not to criticize them."

---

**Q3 — Adversarial: the critical behavior**
> "What is the most critical behavior of this project — the one whose regression would be catastrophic?
> Is there a test that verifies exactly that?"

*If no test:*
> ⚠️ SIGNAL
> The most critical behavior is not covered by a test.
>
> Options:
> A. Brainstorm — we define together how to test it (~15 min)
> B. Continue — I note in kb/risks.md: "critical behavior not tested"
> C. Rephrase — perhaps I misidentified what is critical

---

**Q4 — Adversarial: readability**
> "Can someone other than you pick up this project and understand where everything is in 30 minutes?
> Without you explaining it?"

*If no:*
> "Then bootstrapping the KB has the explicit goal of making that possible.
> We will document the entry points, critical modules, and non-obvious decisions."

---

**Q5 — Neutral: the onboarding objective**
> "What do you want to do with roster on this project?
> What is the first real problem you want to solve?"

→ Orients the install and the first `/roster-run`.

---

**Q6 — Perimeter safety**
> "What parts of the project should I not touch?
> Files, architectures, or non-negotiable dependencies?"

→ Defines the protection scope. Will enter `kb/properties.md` as hard constraints.

### B3. Synthesis before action

```
Here is what I understood about the project:
- State: <clean / identified debt>
- Detected risks: <list>
- Critical behavior: <tested / not tested>
- Non-negotiable constraints: <list>
- Roster objective: <first problem to solve>

Here is what I will install:
- Harness: <agents proposed by recruiter>
- KB draft: <proposed structure>
- Domain skills: <if missing>

Validate before I write anything.
```

Human gate: wait for explicit validation.

### B4. Non-destructive install (after validation)

1. Merge the harness (do not overwrite):
   - If existing team → recruiter Mode 2 (audit + upgrade)
   - If no team → recruiter Mode 1 (fresh, adapted to the project)
2. Propose the KB in the terminal:
   - `kb/spec.md` draft inferred from existing code (README, docs, tests as source)
   - `kb/properties.md` with detected invariants + Q6 constraints
   - `kb/risks.md` with risks identified in B1 and B2-B3
   - Gate: "Here is the KB draft — shall I write it?"
3. If a specific domain is detected without an adapted roster skill:
   - Ask: "Shall I create these skills now via skill-creator?"
   - If yes → spawn `skill-creator` if available (`.claude/agents/skill-creator.md` exists); otherwise manually describe the skill and open a roster issue.
4. Create `skills-meta/friction.jsonl` (empty)
5. Add `skills-meta/` to `.gitignore` if absent
6. Bootstrap episodic memory (non-destructive):
   ```bash
   [ -d memory ] && echo "memory/ exists — skipping" || mkdir -p memory/sessions memory/agents
   ```
   If `memory/` does not exist: create it and write `memory/index.md` (same content as A4 step 9).
   If `memory/` already exists: skip silently.
   Add `kb/.index/` to `.gitignore` if not already present (LanceDB vector index — never committed).
7. Create `briefs/project-intake.md` with project state and first objective
8. Project the harness to runtimes

---

## Brainstorming protocol

Triggered when an adversarial question reveals a fundamental problem and the user chooses option A.

1. Announce the brainstorming subject (1 line)
2. Ask 3 to 5 targeted questions on that specific subject — one at a time
3. Synthesize the answers into an actionable conclusion
4. Write the conclusion in:
   - `kb/risks.md` if it is a risk
   - `kb/spec.md` if it is a scope clarification
5. Resume the interview flow where it was left off

---

## When to Go Back

| Condition | Action |
|---|---|
| Onboarding reveals a deeper structural problem than expected | Stop — surface findings to human before proceeding |
| `.harness/` initialization fails (missing tools, permissions) | Stop — report exact error, do not partially initialize |

## What Next

**Primary path:** `/roster-run` — pipeline is ready, start with a task
**Alternatives:**
- `/roster-intake` — if you already have a task in mind
- `/roster-skill-health` — after first few runs, to capture early friction

> 💡 Run `/roster-skill-health` after your first 3–5 pipeline runs to capture early friction patterns.

## Friction Log

```jsonl
{
  "date": "<ISO-8601>",
  "skill": "roster-init",
  "mode": "<greenfield|onboard>",
  "frictions": ["<friction 1>", "..."],
  "methods": ["<workaround used>"],
  "suggestion_type": "<skill|tool|adapt|agent|null>",
  "suggestion": "<description if suggestion_type non null>",
  "effort_estimate": "<small|medium|large>"
}
```

## Rules

- Never write to the repo before the human gate (validated synthesis)
- Never overwrite an existing file without diff + confirmation
- KB: proposed in the terminal, written only after explicit approval
- Questions: one at a time, never as a list
- If domain is ambiguous for skill creation → ask before spawning `skill-creator`
- The metabolism starts here: friction.jsonl is the first file created
