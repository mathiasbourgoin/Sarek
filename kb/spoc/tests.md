# spoc Tests

## Component Inventory

- `spoc/framework/test/dune`: builds four tests against `spoc_framework`, `alcotest`, and `str` (`spoc/framework/test/dune:1-7`).
- `spoc/ir/test/dune`: builds three tests against `sarek_ir` (`spoc/ir/test/dune:1-3`).
- `spoc/registry/test/dune`: builds one test against `sarek_registry` (`spoc/registry/test/dune:1-3`).
- Framework tests: `test_framework_sig.ml`, `test_typed_value.ml`, `test_device_type.ml`, `test_backend_error.ml`.
- IR tests: `test_sarek_ir_types.ml`, `test_sarek_ir_pp.ml`, `test_sarek_ir_analysis.ml`.
- Registry test: `test_sarek_registry.ml`.

## Per-File Purpose

- `test_framework_sig.ml` checks dimension helpers, capability/device construction, enum distinctness, `exec_arg`, and `run_source_arg` (`spoc/framework/test/test_framework_sig.ml:16-182`).
- `test_typed_value.ml` checks primitives, built-in scalar modules, registry lookup/listing, custom scalar registration, scalar typed values, selected conversions, type-name extraction, and `field_desc` (`spoc/framework/test/test_typed_value.ml:16-250`).
- `test_device_type.ml` checks the manifest alias and representative device records (`spoc/framework/test/test_device_type.ml:14-160`).
- `test_backend_error.ml` uses Alcotest for representative codegen/runtime/plugin formatting and exception/result behavior (`spoc/framework/test/test_backend_error.ml:13-189`).
- `test_sarek_ir_types.ml` checks construction of many IR variants and `vec_length` (`spoc/ir/test/test_sarek_ir_types.ml:16-502`).
- `test_sarek_ir_pp.ml` checks selected string conversions and pretty-printer output prefixes/exact strings (`spoc/ir/test/test_sarek_ir_pp.ml:17-331`).
- `test_sarek_ir_analysis.ml` checks float64 detection across many IR nodes (`spoc/ir/test/test_sarek_ir_analysis.ml:17-409`).
- `test_sarek_registry.ml` checks runtime registry registration/lookup/device-code paths (`spoc/registry/test/test_sarek_registry.ml:16-357`).

## Features and APIs Under Test

- Core type construction and variant distinctness.
- Built-in scalar registration in `Typed_value`.
- Backend error formatting and functor stamping.
- IR float64 analysis for representative composite structures.
- Registry type/record/variant/function lookup and exact module-path keys.

## Test Invariants

- Tests are mostly executable assertions with `print_endline`; only backend error tests use Alcotest assertions and test cases.
- Registry tests mutate global registries without cleanup (`spoc/registry/test/test_sarek_registry.ml:22-282`).
- Typed-value tests mutate `Typed_value.Registry` with a custom scalar and do not reset it (`spoc/framework/test/test_typed_value.ml:124-143`).
- Test names use unique `test_*` registry keys, reducing accidental conflict in one process.

## Potential Test Weaknesses or Bugs

- Many tests use bare `assert`, which can be disabled if compiled with `-noassert`; the backend error suite is more robust because it uses Alcotest checks.
- The registry and typed-value tests depend on global mutable state and have no isolation/reset path.
- Several pretty-printer tests assert only prefixes or string lengths (`spoc/ir/test/test_sarek_ir_pp.ml:143-157`, `248-280`), so malformed output can pass.
- Edge/failure behavior is lightly tested; most tests cover happy paths.

## Performance and Maintainability Risks

- Global registry mutation makes tests order-sensitive if more cases are added with overlapping names.
- Lack of negative tests can allow silent behavior changes in error paths and duplicate handling.
- Test coverage is broad for constructors but shallow for semantic invariants.

## Missing Tests

- Duplicate and ambiguous registry entries.
- Invalid dimensions, zero-device error messages, and unknown type/function failure paths.
- Full `BACKEND` signature compile check with a mock backend.
- Composite typed-value registration and conversion behavior.
- Pretty-printer exact output for complex statements and match patterns.
- IR validation failures once validation exists.
- Float64 analysis for lvalues, native snippets, and helper/type definitions embedded in pretty-printer output.

## Concrete Improvement Candidates

- Convert assertion-style tests to Alcotest across all test files.
- Add registry reset helpers for tests or run mutation-heavy tests in isolated executables with unique names.
- Add negative tests for every `failwith` path.
- Use exact expected strings for pretty-printer outputs where the format is part of the contract.
- Add small property-style tests for recursive analysis consistency, especially nested records, variants, arrays, and vectors.
