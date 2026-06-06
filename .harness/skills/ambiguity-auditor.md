---
name: ambiguity-auditor
description: Audit KB for ambiguity — undefined terms, vague requirements, contradictions, stale content.
version: 1.0.0
---

# Ambiguity Auditor

Scan all knowledge base files and produce a structured report of ambiguity, inconsistency, and staleness issues.

## Steps

### 1. Load KB

- Read `kb/index.md` to enumerate all KB files.
- Read `kb/glossary.md` to build the set of defined terms.
- Read every KB file listed in the index.

### 2. Check: Undefined or Inconsistent Terms

- Extract all domain-specific nouns and noun phrases from each KB file.
- For each term, verify it appears in `kb/glossary.md`.
- Flag terms used but not defined.
- Flag terms defined differently in glossary vs. usage in context (inconsistent usage).

### 3. Check: Vague Requirements

Scan for patterns indicating unquantified or untestable requirements:

- Subjective qualifiers: "should be fast", "handle errors gracefully", "in a timely manner", "as needed", "if possible", "reasonable", "appropriate"
- Missing thresholds: "low latency" (how low?), "high availability" (what SLA?), "scalable" (to what load?)
- Weasel phrases: "generally", "typically", "in most cases", "may", "might"

For each match, flag and recommend a quantified replacement.

### 4. Check: Contradictions Between Files

- Cross-reference claims across all KB files.
- Specifically compare: `spec.md` vs `properties.md`, `architecture.md` vs `spec.md`, any file vs `glossary.md`.
- Flag: same concept described with conflicting behavior, conflicting constraints, or conflicting ownership.

### 5. Check: Missing Cross-References

- When a KB file mentions a concept that has its own KB entry or glossary definition, verify a cross-reference exists (link, explicit "see X", or inline definition).
- Flag missing cross-references.

### 6. Check: Stale Content

- Check `last-updated` frontmatter in each KB file. Flag files older than 30 days.
- Check for references to components, functions, or files that no longer exist in the codebase (use `find` / `grep` to verify).
- Flag dead references.

### 7. Cross-Spec Entity Consistency (conditional)

```bash
ls specs/*.md 2>/dev/null
```

If specs found:
1. For each spec file, extract all lines under `## Entities`
2. Build a map: entity_name → {definition, source_file}
3. For any entity_name appearing in more than one spec with different definitions:
   - Report as CRITICAL finding: "Entity `<name>` defined differently in <file1> vs <file2>"
   - Include both definitions in the report
4. For any AC in a new spec referencing an entity not defined in its `## Entities` section:
   - Report as WARNING: "AC references undefined entity `<name>`"

Write findings to `kb/reports/ambiguity-report.md` under new section `## Cross-Spec Entities`.

### 8. Generate Report

Write the report to `kb/reports/ambiguity-report.md` using this format:

```markdown
---
auditor: ambiguity-auditor
date: <today YYYY-MM-DD>
status: N critical, N warnings, N info
---

## Critical

### [C1] <Issue title>
- **Location**: <file>, <line or section>
- **Issue**: <description>
- **Recommendation**: <specific fix>

## Warnings

### [W1] <Issue title>
- **Location**: <file>, <line or section>
- **Issue**: <description>
- **Recommendation**: <specific fix>

## Info

### [I1] <Issue title>
- **Location**: <file>, <line or section>
- **Note**: <description>
```

**Severity mapping:**
- **Critical**: Contradictions, undefined terms used in spec/properties, vague requirements in spec.
- **Warning**: Missing cross-references, inconsistent term usage, vague requirements outside spec.
- **Info**: Stale content, minor style issues, suggested improvements.

## Rules

- Report **every** finding — do not skip issues to keep the report short.
- Never auto-fix KB content during an audit — only report.
- Always create the `kb/reports/` directory if it doesn't exist.
- If `kb/glossary.md` doesn't exist, flag that as Critical #1.
