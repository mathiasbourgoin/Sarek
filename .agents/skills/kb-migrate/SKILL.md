---
name: kb-migrate
description: Audit, clean, reorg, and migrate an existing KB to the current schema — idempotent, human-gated at each phase.
version: 1.0.0
---

# KB Migrate

You perform a structured, human-gated migration of an existing project KB to the current roster schema. Five sequential phases: audit, cleanup, reorg, frontmatter migration, verify.

**Idempotency guarantee:** running this skill twice on the same KB produces no additional changes. Files with `schema-version: 2` in frontmatter are skipped in Phase D.

## Pre-conditions

```bash
[ -d kb ] && echo "KB present" || echo "KB absent — nothing to migrate"
```

If no `kb/` directory: report "No KB found. Run `/roster-init` to bootstrap one." and stop.

## Steps

### Phase A — Audit

Run all available auditors on the existing KB:

1. Run `ambiguity-auditor` on all KB files. Collect: undefined terms, vague quantifiers, contradictions.
2. Run `spec-compliance-auditor` if codebase is present. Collect: spec vs code divergences.
3. Run `check-kb-links`: `node scripts/check-kb-links.js` (or `npm run check:kb-links`). Collect: broken relative links.

Produce a **Findings Summary**:

```
## Audit Findings

### Broken links (N)
- <file>: [link] → <target> (not found)

### Ambiguity issues (N)
- <file>: <issue>

### Spec compliance issues (N)
- <file>: <issue>

### Migration candidates (files without schema-version: 2)
- <file>: status=<old-value>
```

Present findings to the human. **Gate:** proceed to Phase B only after human acknowledges.

---

### Phase B — Cleanup

For each finding from Phase A:
- Broken link: propose the correct target (search for similar filenames) or mark as "file missing — create or remove link". Human approves per item. Apply approved fixes.
- Ambiguity issue: propose precise language replacement. Human approves. Apply.
- Spec compliance: present the divergence, ask: "Fix code or amend spec?" Do not resolve unilaterally.

After all items are resolved or explicitly deferred:

```
## Cleanup Summary
Applied: N fixes
Deferred: M items (listed)
```

**Gate:** human confirms cleanup is complete.

---

### Phase C — Reorg

Compare current KB structure against the schema tier (minimal/standard/large) appropriate for the project size:

```
Minimal  (<5 files):  index.md, spec.md, glossary.md
Standard (5-15 files): + architecture.md, properties.md, decisions/, reports/
Large    (>15 files):  + modules/, runbooks/
```

Identify:
- Files present but not in `index.md` (orphaned)
- Sections listed in `index.md` but no file (broken index entries)
- Files that belong in a subdirectory (e.g., raw ADRs not in `decisions/`)

Propose restructuring as a diff: "Move X to Y", "Add index entry for Z", "Create missing file W". Human approves the reorg plan. Apply approved moves/creates.

**Gate:** human confirms reorg is acceptable.

---

### Phase D — Frontmatter Migration

For each KB file (excluding `reports/`):

1. Read the file's frontmatter.
2. **If `schema-version: 2` is already present → skip this file (idempotent no-op).**
3. Otherwise, determine changes:
   - Map `status` value: `draft` → `live-doctrine`, `reviewed` → `live-doctrine`, `stale` → `historical`
   - If `status` is already a new-format value (`live-doctrine`, `superseded`, `historical`, `derived`) → keep it, still add `schema-version: 2`
   - Add `schema-version: 2`
   - Add `superseded-by: null` if file is a spec file (`spec.md`, `architecture.md`, `properties.md`, `glossary.md`)
   - Add `supersedes: null` if file is in `decisions/`

Produce a migration plan listing every file and change before writing anything:

```
Migration plan:
  kb/spec.md: status draft → live-doctrine, add schema-version: 2, add superseded-by: null
  kb/decisions/001-auth.md: status reviewed → live-doctrine, add schema-version: 2, add supersedes: null
  kb/reports/audit-2025-01.md: (skip — reports/ excluded)
  ...
```

**Gate:** human approves the migration plan before any writes.

Apply changes file by file. Update `last-updated` to today's date on each modified file.

---

### Phase E — Verify

1. Re-run `check-kb-links`. Must report 0 broken links.
2. Re-run `ambiguity-auditor` on migrated files. Report any new issues (there should be none from frontmatter-only changes).
3. Verify all migrated files have `schema-version: 2`.

Report:

```
## Migration Complete

Files migrated: N
Files skipped (already current): M
Broken links remaining: 0
Ambiguity issues introduced: 0
```

If any check fails, report specifically and stop — do not mark migration complete.

## Rules

- Never change KB content (spec text, property definitions, glossary terms) — frontmatter only in Phase D
- Never apply fixes from Phase B, C, or D without explicit human approval
- Phase D is idempotent: re-running it on an already-migrated KB must be a no-op
- Do not delete any file — only move (Phase C) with human approval
- Deferred items from Phase B must be listed explicitly, never silently dropped
