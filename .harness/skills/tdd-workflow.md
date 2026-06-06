---
description: Run TDD cycle — write failing test, implement, refactor, verify coverage.
version: 1.0.0
---

# TDD Workflow

You guide the user through a strict red-green-refactor cycle for the feature or fix described in $ARGUMENTS.

## Language Detection

Before starting, detect the project's test framework:

| Marker | Runner | Command |
|--------|--------|---------|
| `package.json` with `vitest` | Vitest | `npx vitest run` |
| `package.json` with `jest` | Jest | `npx jest` |
| `pytest.ini` / `pyproject.toml` / `conftest.py` | pytest | `pytest` |
| `Cargo.toml` | cargo test | `cargo test` |
| `dune-project` | dune test | `dune runtest` |
| `go.mod` | go test | `go test ./...` |
| `mix.exs` | ExUnit | `mix test` |
| `*.cabal` / `stack.yaml` | HSpec/Tasty | `cabal test` or `stack test` |

If no marker found, ask the user which framework to use.

Store the detected runner and command as $RUNNER and $TEST_CMD for all subsequent steps.

## Steps

### 1. RED — Write Failing Test

- Parse $ARGUMENTS to understand the desired behavior.
- Write a test that asserts the expected behavior. The test must compile/parse but **fail** when run.
- Run `$TEST_CMD`. Confirm the new test fails. If it passes, STOP — the behavior already exists or the test is wrong. Ask the user to clarify.

### 2. GREEN — Minimal Implementation

- Write the **minimum** code to make the failing test pass. No extra features, no premature abstractions.
- Run `$TEST_CMD`. Confirm the new test passes AND all existing tests still pass.
- If any test fails, fix only the implementation (never the test).

### 3. REFACTOR — Improve

- Review both the test and implementation for:
  - Duplication (DRY)
  - Naming clarity
  - Unnecessary complexity
  - Extract helpers if test setup is repeated
- Apply improvements. Run `$TEST_CMD` to confirm everything stays green.

### 4. VERIFY — Coverage Check

- Run coverage if the framework supports it:
  - Jest/Vitest: `$TEST_CMD --coverage`
  - pytest: `pytest --cov`
  - cargo: `cargo tarpaulin` (if installed) or `cargo llvm-cov`
  - go: `go test -cover ./...`
- Report coverage. If below 80% (or user-specified threshold), identify untested paths and loop back to Step 1 for additional cases.

### 5. REPORT

Summarize: what behavior was added, how many tests pass, coverage %, and any refactoring done.

## Rules

- **Never** write implementation before a failing test exists.
- **Never** modify a test to make it pass — fix the implementation instead.
- If stuck, simplify the test to a smaller assertion and iterate.
- Run the **full** suite after every change, not just the new test.
- Each cycle adds exactly one behavior. For multiple behaviors, repeat the cycle.
