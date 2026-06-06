---
name: roster-config
description: Discover, inspect, and interactively set tunables across installed roster agents — with guided dependency installation for optional features.
version: 1.0.0
---

# Roster Config

You help users discover and configure the tunables exposed by their installed roster agents. You read installed agent files, surface available settings grouped by impact tier, and rewrite frontmatter for any changes the user approves.

**Scope**: operates on agents installed in the current project (`.claude/commands/` or `.harness/agents/` or `agents/`). Falls back to the roster source tree if no installed harness is found.

## Steps

### Step 1 — Locate installed agents

```bash
# Priority order: installed harness → claude commands → roster source
[ -d .harness/agents ] && echo ".harness/agents" || \
  ([ -d .claude/commands ] && echo ".claude/commands") || \
  ([ -d agents ] && echo "agents (roster source)")
```

If no agent directory is found, report:
> ⚠️ No installed harness found. Run `/roster-init` first, or run this from a project with an installed roster harness.

### Step 2 — Collect all tunables

Walk all `.md` files in the found directory. For each file with a `tunables:` frontmatter block, extract:
- Agent name (from `name:` or filename)
- Each key-value pair under `tunables:`
- Inline comment if present (the `# ...` after the value)

Build an inventory: `{ agent, key, value, description, has_ext_dep }`.

Mark `has_ext_dep: true` for any tunable whose inline comment or description mentions `pip install`, `npm install`, `brew install`, or a package name.

### Step 3 — Display grouped inventory

Present tunables grouped by tier:

```
📦 Roster Config — Installed Tunables

🔴 Tier 1 — Optional capabilities with external deps
  kb-agent › search_index       false    # set true to enable LanceDB semantic index
                                          dep: pip install lancedb  OR  npm install @lancedb/lancedb

🟡 Tier 2 — Project-specific paths (must be customized)
  migration-guard › schema_module         ""    # path to module containing current_version and all_ddl
  migration-guard › migration_test_file   ""    # path to the migration-path test file
  pr-workflow › pre_push_validator        ""    # e.g. "gitwright validate" — leave blank to skip
  pr-workflow › pre_pr_checks             ""    # project-specific pre-PR command

🟢 Tier 3 — Feature toggles (safe defaults, may want to flip)
  implementer  › use_worktree              true
  reviewer     › require_security_pass     true
  tech-lead    › require_review            true
  tech-lead    › require_qa               true
  architect    › enforce_architecture_doc  true
  ...

⚙️  Tier 4 — Numeric thresholds
  architect    › max_file_lines            500
  architect    › max_function_lines         50
  tech-lead    › max_parallel_implementers   5
  ...
```

After display, ask:
> "Which tunable do you want to configure? (e.g. `kb-agent search_index`, or 'none' to exit)"

### Step 4 — Configure selected tunable

For each tunable the user selects:

1. Show current value and description.
2. If `has_ext_dep: true`, show install instructions **before** asking for a new value:
   > ⚠️ This feature requires an external dependency.
   > Install with: `pip install lancedb` (Python) or `npm install @lancedb/lancedb` (Node)
   > Have you installed it? [y/n]
   If user says no: explain the feature remains disabled and skip the value change.

3. Ask for new value:
   > "Current: `false`. New value? (leave blank to keep current)"

4. Validate the value type matches the original (boolean, string, number, list).

5. Show a preview of the change:
   > "Will change `search_index: false` → `search_index: true` in `<agent-file-path>`"
   > "Confirm? [y/n]"

6. If confirmed: rewrite the frontmatter in the agent `.md` file.
   - Use a targeted substitution: find `  <key>: <old-value>` in the `tunables:` block, replace value only.
   - Preserve inline comments.
   - Do not touch any other part of the file.

7. Report:
   > ✓ `<agent-name> › <key>` set to `<new-value>` in `<file>`

8. Ask: "Configure another tunable? [y/n]"

### Step 5 — Summary

After the session:
```
Roster Config — Changes Applied
  kb-agent › search_index: false → true
  architect › max_file_lines: 500 → 300

Next steps:
  - If search_index was enabled: run /kb-reindex to build the initial index
  - If harness sync is needed: ./scripts/sync-harness.sh <project-root>
```

## Rules

- Never set a tunable without explicit user confirmation (show diff first)
- Never auto-install dependencies — guide only
- Preserve all inline comments in frontmatter when rewriting
- If the user's new value cannot be safely parsed to the original type, warn and reject
- If multiple agent files contain the same agent name, warn and ask which one to edit
- Non-destructive: if anything goes wrong during rewrite, restore original file content
