---
name: spec-compliance-auditor
description: Audit implementation against kb/spec.md — flag unimplemented spec items and behavioral divergence.
version: 1.0.0
---

# Spec Compliance Auditor

Compare the implementation against a spec file to verify every specified behavior is implemented and every implemented behavior is specified.

## Input Contract

- If `$ARGUMENTS` contains a file path to a spec: use that file as the spec source
- Default (no `$ARGUMENTS`): read `kb/spec.md`
- Fail gracefully if neither exists: report "No spec source available" and exit

## Steps

### 1. Extract Testable Claims

```bash
SPEC_PATH="${ARGUMENTS:-kb/spec.md}"
[ -f "$SPEC_PATH" ] || { echo "No spec at $SPEC_PATH — skipping"; exit 0; }
```

- Read the spec file at `$SPEC_PATH` in full.
- Extract every concrete, testable claim. A testable claim is a statement that can be verified by reading code or running a test. Examples:
  - "The API returns 404 when the resource is not found"
  - "Passwords are hashed with bcrypt, cost factor 12"
  - "The retry limit is 3 attempts with exponential backoff"
- Ignore aspirational or process-oriented statements ("we aim to...", "the team will...").
- Number each claim for reference: S1, S2, S3, etc.

### 2. Verify Each Claim

For each testable claim:

1. **Locate** the relevant implementation code (search for function names, types, constants mentioned in the claim).
2. **Read** the code and determine if the behavior matches the spec.
3. **Check tests** — does a test exist that verifies this specific claim? Note the test file and name.
4. **Classify**:

| Status | Meaning |
|--------|---------|
| **PASS** | Code matches spec, test exists |
| **PARTIAL** | Code matches spec, no test covering it |
| **DIVERGE** | Code behaves differently than spec states |
| **MISSING** | No implementation found for this spec item |

### 3. Check for Unspecified Implementations

- Scan the codebase for public APIs, exported functions, and user-facing features.
- For each, verify it has a corresponding spec entry.
- Flag features implemented but not in spec as "unspecified" — these are either spec gaps or unauthorized features.

### 4. Generate Report

Write to `kb/reports/spec-compliance-report.md`:

```markdown
---
auditor: spec-compliance-auditor
date: <today YYYY-MM-DD>
status: N critical, N warnings, N info
coverage: X/Y claims verified (Z%)
---

## Compliance Matrix

| Claim | Status | Location | Test | Notes |
|-------|--------|----------|------|-------|
| S1: <short desc> | PASS/PARTIAL/DIVERGE/MISSING | <file:line> | <test file> | <notes> |

## Critical

### [C1] <Issue title>
- **Spec claim**: S<N> — "<quote from spec>"
- **Actual behavior**: <what the code does>
- **Location**: <file>, <line>
- **Recommendation**: <fix code or amend spec via discussion>

## Warnings

### [W1] <Issue title>
- **Details**: <description>
- **Recommendation**: <action>

## Info

### [I1] Unspecified feature: <name>
- **Location**: <file>
- **Note**: Implemented but not in spec. Consider adding to spec or removing.
```

**Severity mapping:**
- **Critical**: DIVERGE (code contradicts spec), MISSING (spec item not implemented).
- **Warning**: PARTIAL (implemented but untested), unspecified features in critical paths.
- **Info**: Unspecified features in non-critical paths, minor spec ambiguities discovered.

## Rules

- **Spec is the source of truth.** When code diverges, recommend fixing the code — never silently update the spec.
- If a spec claim is ambiguous, flag it as Info and note the ambiguity — do not guess intent.
- Run existing tests where possible to confirm behavior rather than relying solely on code reading.
- Create `kb/reports/` directory if it doesn't exist.
- If `kb/spec.md` doesn't exist and no `$ARGUMENTS` path is given, report "No spec source available" and skip (exit 0).
