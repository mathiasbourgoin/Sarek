---
name: code-quality
description: Universal code quality limits — file length, function length, nesting, naming, dead code.
scope: global
category: style
version: 1.0.0
---

# Universal Code Quality Limits

## Size Limits

- **Max file length:** 500 lines (configurable). If a file exceeds this, split it by responsibility.
- **Max function length:** 50 lines (configurable). If a function exceeds this, extract sub-functions.
- **Max nesting depth:** 4 levels. Use early returns, guard clauses, or extracted functions to flatten.

## Structure

- No deep inheritance hierarchies. Prefer composition over inheritance.
- Functions should do one thing. If a function name requires "and", it does too much.

## Naming

- Names must be descriptive and reveal intent.
- No single-letter variables except: loop counters (`i`, `j`, `k`), coordinates (`x`, `y`, `z`), and well-known mathematical conventions.
- Avoid abbreviations unless they are universally understood in the domain (e.g., `ctx`, `cfg`, `db`).

## Dead Code

- No dead code. If code is commented out, delete it. Version control exists for a reason.
- No TODO comments older than the current PR. File an issue instead and reference the issue number if needed.
- No unreachable code paths. If a branch can never execute, remove it.
