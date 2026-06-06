---
name: kb-update
description: Update knowledge base — sync KB files with recent code changes without weakening specs.
version: 1.1.0
---

# KB Update

Synchronize the project knowledge base with recent changes. This is the standalone invocable version of the KB agent's update mode.

## Steps

### 1. Read KB Structure

- Read `kb/index.md` to understand the KB layout and file purposes.
- Identify all KB files: `spec.md`, `properties.md`, `glossary.md`, `architecture.md`, and any domain-specific files listed in the index.

### 2. Identify Recent Changes

- Run `git diff HEAD~5..HEAD --name-only` to find recently changed source files (adjust range if user specifies).
- Run `git diff HEAD~5..HEAD` for the actual diffs.
- Identify which KB-relevant concepts were touched (new types, changed APIs, modified invariants, renamed terms).

### 3. Compare Code Against KB

For each changed concept, classify the delta:

| Delta Type | Action |
|-----------|--------|
| Code **contradicts** KB spec | **FLAG AS ERROR**. Do NOT update KB. Report: "Code at `<file>:<line>` violates spec: `<quote>`. Fix the code or open a spec amendment discussion." |
| Code **extends** KB (new feature, new term) | Add entries to relevant KB files. Update `kb/index.md` if new files created. |
| Code **refines** existing concept | Update the KB entry with more precise language. Preserve or strengthen existing properties. |
| KB entry has no corresponding code | Flag as potentially stale. Do not remove — ask the user. |

### 4. Apply Updates

- For each KB file modified, update the frontmatter `last-updated` field to today's date.
- Add cross-references between related entries.
- Ensure new terms are added to `kb/glossary.md` with definitions.

### 5. Run Ambiguity Check

- After all updates, invoke the ambiguity-auditor checks on modified KB files:
  - No undefined terms (all terms in glossary)
  - No vague quantifiers ("fast", "many", "soon")
  - No contradictions introduced
- Fix any issues found or flag them for the user.

### 6. Report

Summarize: files updated, entries added/modified, contradictions found, ambiguity issues.

### 7. Trigger reindex (conditional)

```bash
[ -d kb/.index ] && echo "index: present" || echo "index: absent"
```

If `kb/.index/` exists: invoke `/kb-reindex` in incremental mode, passing the list of files modified in Steps 4 and 5. This keeps the search index in sync without a full reindex.

If `kb/.index/` does not exist: skip silently — the index is opt-in.

## Rules

- **Never** weaken `properties.md` — properties can only be strengthened or added, never relaxed.
- **Never** change `spec.md` to match code — spec is the source of truth. If code diverges, flag it.
- **Never** delete KB entries without explicit user approval.
- **Never** add speculative content — only document what exists in code or is stated in spec.
- All KB edits must be traceable to a specific code change or user request.
