---
name: code-quality-auditor
description: Audit code quality against KB-defined properties, invariants, and naming conventions.
version: 1.0.0
---

# Code Quality Auditor

Check implementation against knowledge base properties and conventions. Uses only built-in tools (grep, wc, read) — no external linters required.

## Steps

### 1. Load KB References

- Read `kb/properties.md` for invariants, constraints, and quality thresholds.
- Read `kb/glossary.md` for canonical naming conventions and term definitions.
- Read `kb/spec.md` for architectural boundaries and module responsibilities.

### 2. Check: Function Size

- Scan source files for function/method definitions.
- Measure line count per function (use grep + wc approach).
- Flag functions exceeding the threshold defined in `properties.md` (default: 50 lines if not specified).
- Exclude test files from size checks unless properties.md says otherwise.

### 3. Check: DRY Violations

- Search for duplicated code blocks (identical or near-identical sequences of 5+ lines).
- Look for repeated patterns: duplicated error handling, copy-pasted validation logic, repeated configuration blocks.
- Flag each duplication with both locations.

### 4. Check: Naming Consistency with Glossary

- For each term in `kb/glossary.md`, search the codebase for variant spellings, abbreviations, or synonyms.
- Example: if glossary defines "Transaction", flag code using "tx", "txn", "trans" inconsistently.
- Check that type names, function names, and module names use glossary-canonical forms.

### 5. Check: Invariant Preservation

- For each invariant in `kb/properties.md`, verify the code maintains it:
  - If an invariant says "X is never null", search for code paths where X could be null.
  - If an invariant says "Y is always called before Z", trace call sequences.
  - If an invariant says "all inputs are validated", check for unvalidated entry points.
- This is best-effort static analysis — flag suspicious patterns, don't claim proof.

### 6. Check: Error Handling

- Scan for empty catch blocks, swallowed errors, and generic catch-all handlers.
- Flag functions that can throw/error but whose callers don't handle failures.
- Check consistency with error handling patterns defined in properties.md.

### 7. Generate Report

Write to `kb/reports/code-quality-report.md`:

```markdown
---
auditor: code-quality-auditor
date: <today YYYY-MM-DD>
status: N critical, N warnings, N info
---

## Critical

### [C1] <Issue title>
- **Location**: <file>, <line>
- **Issue**: <description>
- **KB reference**: <properties.md section or glossary entry>
- **Recommendation**: <specific fix>

## Warnings

### [W1] <Issue title>
- **Location**: <file>, <line>
- **Issue**: <description>
- **KB reference**: <properties.md section or glossary entry>
- **Recommendation**: <specific fix>

## Info

### [I1] <Issue title>
- **Location**: <file>, <line>
- **Note**: <description>
```

**Severity mapping:**
- **Critical**: Invariant violation, broken naming convention in public API, security-relevant quality issue.
- **Warning**: Function too long, DRY violation, inconsistent naming in internal code, empty catch block.
- **Info**: Style suggestions, minor naming variants, refactoring opportunities.

## Rules

- Always reference the specific KB entry that defines the violated property.
- Never suggest fixes that would violate other KB properties.
- Use only grep, wc, read, and glob for checks — no external tools.
- Create `kb/reports/` directory if it doesn't exist.
- If `kb/properties.md` doesn't exist, report that as Critical #1 and skip invariant checks.
