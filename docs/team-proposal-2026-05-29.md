# SPOC Harness Update — 2026-05-29

Scope (selected by Mathias): **Bump + re-adapt + add OCaml agents.** SoT: `.harness/`,
runtime entrypoints re-rendered via `sync-harness.sh`.

## tl;dr

- Bump 10 installed agents to current roster versions; preserve all local tunables.
- Bump 3 drifting skills (`kb-update`, `improvement-loop`, `improvement-loop-planner`).
- Install `planner` agent (now expected by `tech-lead` v1.9.0 Mode A/B execution).
- Install `ocaml-implementer` and `ocaml-dune-specialist` as OCaml-specialised siblings of the generic `implementer` and `architect`. Generic ones stay; lead routes by file path.
- Install `diagnostic-interview` rule (front-door governance for fuzzy intake; referenced by upgraded `tech-lead` and `recruiter`).
- Copy `patterns/ocaml.md` from the roster into `.claude/patterns/`. Flag missing `c.md`, `javascript.md`, `shell.md` as a follow-up (do not block this update on creating them).
- Update `harness.json` manifest. Run `sync-harness.sh` to re-render `.claude/` and `.agents/skills/recruit/`.

## Version bump table

| Agent | Installed | Roster | Action |
|---|---|---|---|
| recruiter | 2.2.0 | 2.5.2 | merge: keep `tunables.roster_repo=mathiasbourgoin/agent-roster`, take new body |
| tech-lead | 1.6.0 | 1.9.0 | merge: keep `tunables.max_parallel_implementers=5`, take new body (SPAWN REQUEST format, planner awareness) |
| implementer | 1.1.0 | 1.3.0 | merge tunables |
| reviewer | 1.2.0 | 1.4.0 | merge tunables |
| qa | 1.1.0 | 1.3.0 | merge tunables |
| architect | 1.3.0 | 1.5.0 | merge tunables (keep code-quality thresholds 500/50/0.15) |
| kb-agent | 2.1.0 | 2.4.0 | merge tunables (`kb_dir=kb`) |
| project-auditor | 1.0.0 | 1.1.0 | merge tunables |
| red-team-auditor | 1.0.0 | 1.1.0 | merge tunables (`audit_mode`, scope, report_dir, etc.) |
| performance-monitor | 1.1.0 | 1.2.0 | merge tunables |

| Skill | Installed | Roster | Action |
|---|---|---|---|
| kb-update | 1.0.0 | 1.1.0 | overwrite |
| improvement-loop | 1.0.0 | 1.1.0 | overwrite |
| improvement-loop-planner | 1.0.0 | 1.1.0 | overwrite |
| tdd-workflow | 1.0.0 | 1.0.0 | no-op |
| git-conventions | 1.0.0 | 1.0.0 | no-op |

## Additions

- **`planner` 1.2.0** (`agents/management/planner.md`). Required by tech-lead v1.9.0's pipeline. Receives the research brief from tech-lead and decomposes it into per-implementer sub-briefs. Output is consumed by implementer (or ocaml-implementer) before any code lands.
- **`ocaml-implementer` 1.2.0** (`agents/backend/ocaml-implementer.md`) installed **as upstream reference only**. Its body is wired for a different OCaml archetype (eio_posix + Cohttp + SQLite/Caqti) that does not match SPOC. The lead does NOT route to it.
- **`spoc-ocaml-implementer` 0.1.0** — SPOC-flavored fork of the upstream, created locally. Routed for `*.ml/*.mli/*.mly/*.mll` and paths under `sarek/`, `spoc/`, `sarek-*/`. Knows local opam switch, ctypes FFI symmetry, Sarek PPX rewriters, license-header script, ocamlformat. Carries `tunables.upstream_ref = ocaml-implementer` and `tunables.upstream_known_version = 1.2.0`; on first use per session it diffs the upstream version and prompts for manual re-derivation if drifted. Mode 4 PR for this back to the roster is a separate follow-up.
- **`ocaml-dune-specialist` 1.2.0** (`agents/specialist/ocaml-dune-specialist.md`). Standby specialist — only spawned when dune module boundaries, public name collisions, ppx rewriter wiring, or library packaging is in scope. Architect remains the generic structural guardian.
- **`diagnostic-interview` rule** (`rules/governance/diagnostic-interview.md`). Front-door protocol for fuzzy / high-stakes intake. Wired into upgraded tech-lead and recruiter automatically (no separate patching needed for new agent bodies).
- **`patterns/ocaml.md`** copied into `.claude/patterns/ocaml.md`. Roster has no `c.md` / `javascript.md` / `shell.md`; those are flagged for a later Mode 4 pass.

## Pipeline topology after this update

```
[Mathias] -> tech-lead (research + brief)
          -> [Mathias validates brief]
          -> planner (sub-briefs, per affected slice)
          -> [Mathias validates decomposition]
          -> implementer  | spoc-ocaml-implementer   (chosen by lead, by file paths)
                -> reviewer -> qa
          -> tech-lead (merge decision)
          -> [Mathias approves merge]

  (ocaml-implementer is installed as upstream reference; not routed to. The
   spoc-ocaml-implementer fork checks it for drift on first use and prompts.)

Standby specialists (spawned only when their slice is in scope):
  - architect (cross-cutting structural review)
  - ocaml-dune-specialist (dune/ppx/library boundaries)
  - red-team-auditor (security-sensitive surfaces: FFI, ctypes, runtime, ppx)
  - performance-monitor (benchmark/CI regressions)
  - project-auditor (kb refresh / cold audit)
  - kb-agent (kb/ delta after merge)
```

Execution remains **Mode B (human-mediated sequential)** by default — Mathias spawns
each agent with the prior stage's context. Mode A (parallel batch) is only used when the
lead has produced fully disjoint sub-briefs and the human explicitly batches them.

## Layer 3 — Lead and adjacency patches

These are NOT separate frontmatter edits; they are bodies inside the newly-bumped agent
files in the roster (already account for planner + ocaml-* siblings). Verification step
during install:

1. `tech-lead` v1.9.0 body: contains the SPAWN REQUEST block referencing `planner`. ✓
2. `tech-lead` v1.9.0 body: routes implementation by stack/path, accepting `ocaml-implementer` as a valid implementer alias. (Verify after writing.)
3. `implementer` v1.3.0 body: declares the OCaml carve-out. (Verify after writing.)
4. `architect` v1.5.0 body: declares ocaml-dune-specialist as a delegate for dune-specific concerns. (Verify after writing.)

If any of (2)–(4) is not in the roster body, I patch the installed file with an explicit
**Routing** section and surface the patch in the diff.

## Dependencies

| Tool | Type | Needed by | Status |
|---|---|---|---|
| gh | cli | recruiter | available (verify with `gh auth status` if not done recently) |
| git | cli | project-auditor, red-team-auditor | available |
| ripgrep | cli | project-auditor, red-team-auditor | optional, already in dev env |
| semgrep, codeql | cli | red-team-auditor | optional, no change |

No new dependencies introduced by this update.

## Tunables to preserve (explicit list)

- `recruiter.tunables.roster_repo` — **drop the local override** (`mathiasbourgoin/agent-roster`) and take the roster default `mathiasbourgoin/roster`. Verified: the local checkout's git remote is `git@github.com:mathiasbourgoin/roster.git`; the override was stale (directory name vs. repo name).
- `tech-lead.tunables.max_parallel_implementers = 5`
- `architect.tunables.max_file_lines = 500, max_function_lines = 50, max_duplication_threshold = 0.15, enforce_architecture_doc = true`
- `kb-agent.tunables.kb_dir = kb`
- `project-auditor.tunables.kb_dir = kb` and the rest of its policy block
- `red-team-auditor.tunables.*` (the project-adaptive audit policy block in full)
- `performance-monitor.tunables.max_optimization_candidates = 5, require_baseline = true`
- `qa.tunables.run_full_suite = true, include_manual_checks = true`
- `implementer.tunables.use_worktree = true, run_tests_before_handoff = true, prefer_small_commits = true`

## Binding decisions (from validation quiz)

- **Routing:** lead routes by file path. `ocaml-implementer` for `sarek/`, `spoc/`, `sarek-*/`, `*.ml/.mli/.mly/.mll`. Generic `implementer` for C, JS, shell, docs, CI, Dockerfile.
- **Merge order:** single-write per file. Read roster body + (sparse) local tunables in memory, splice, write merged file once. No on-disk intermediate state where the merged result is incomplete.
- **`recruiter.roster_repo`:** drop the stale override, take the roster default `mathiasbourgoin/roster`.

## Order of operations

1. Bump 10 agents in `.harness/agents/` (single-write merge, preserve listed tunables except `recruiter.roster_repo`).
2. Verify layer-3 routing language in tech-lead, implementer, architect new bodies. Patch if missing.
3. Write `planner`, `ocaml-implementer`, `ocaml-dune-specialist` to `.harness/agents/`.
4. Write `diagnostic-interview` to `.harness/rules/`.
5. Bump 3 skills in `.harness/skills/`.
6. Copy `patterns/ocaml.md` to `.claude/patterns/ocaml.md`. (`.claude/patterns/` does not exist yet — create it.)
7. Update `.harness/harness.json` (version → 1.1.0, agent/skill/rule lists refreshed).
8. Run `/home/mathias/dev/agent-roster/scripts/sync-harness.sh /home/mathias/dev/SPOC` to re-render `.claude/` and `.agents/skills/recruit/`.
9. Diff summary printed for review.

## What this update intentionally does NOT do

- Does not install `harness-builder`, `skill-creator`, `pr-workflow`, `expert-debugger`, `context-manager`, `mcp-vetter`, `tool-provisioner`, `config-migrator`, `migration-guard`, or roster pipeline skills (`roster-*`). Those are useful but out of scope for "bump + re-adapt + OCaml" and would push past `max_team_size: 10` in spirit.
- Does not create new pattern files for C, JavaScript, Shell. Roster lacks them; creating + PR-ing them is a separate Mode 4 pass.
- Does not run `/recruit govern`. The governance rules are already installed and just gain `diagnostic-interview` here.
- Does not modify `kb/`, `CLAUDE.md`, or any code under `sarek/`, `spoc/`, `sarek-*/`.
