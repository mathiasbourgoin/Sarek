# Reviewer Sub-brief — pure-codegen-rollout PR-5a (pure stdlib metadata lib)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-5 of 5, part a)

## What was implemented
`%sarek_intrinsic` type-size derivation made Ctypes-free (derive size from sarek_type; `ctype`
field optional); new pure `sarek_stdlib_meta` lib registers stdlib intrinsic metadata
(name/sarek-type/device-template/size) FFI-free; FFI `sarek_stdlib` reduced to ctype+native/Vector
on top of it; a proof exe links `sarek_frontend + sarek_stdlib_meta` (no FFI) and asserts the
registry resolves `Float32.sin` etc.

## Load-bearing GO/NO-GO hinges
1. **`sarek_stdlib_meta` is FFI-free AND populates the registry.** Read its dune `(libraries …)` —
   NO `spoc_core`/`ctypes`. The proof exe must link only pure libs and PASS (registry contains
   `Float32.sin` with correct type). `dune build sarek_stdlib_meta` standalone succeeds. If FFI
   leaked or the registry isn't populated FFI-free → NO-GO (this is the entire point).
2. **Byte-identical goldens.** `git diff origin/main -- sarek/tests/codegen_golden/` EMPTY → NO-GO if not.
3. **Native/interpreter identical.** `test_math_intrinsics --native` and `--interpreter` pass — the
   FFI execution path (ctype/Vector) must be intact after the split. → NO-GO if regressed.

## Verify
- **No double/divergent registration:** if both `sarek_stdlib_meta` and `sarek_stdlib` register, the
  data must be identical (else the registry depends on link order). Prefer single-source (meta only).
  Confirm which lib registers what and that there's no conflicting `ti_size`/type for the same name.
- **Size derivation correctness:** the sarek-type→size map (TFloat32→4, TInt32→4, TFloat64→8,
  TInt64→8, TBool→4, TChar→1, Custom→?) matches the OLD `Ctypes.sizeof` values for every registered
  type. A wrong size would silently corrupt Vector marshalling — check against the prior ctype sizes.
- **Coverage:** confirm which intrinsic modules are in the pure meta lib. If the implementer
  escalated to a partial set (per the brief's clause), confirm the FFI-free proof covers exactly the
  claimed set and the rest still works via the FFI path. Flag the deferred set as tracked follow-up.
- **All consumers build:** plugins, e2e, native exec, benchmarks. Full `dune build` (only `-lnvrtc`).
  `@sarek-vulkan/all` builds. A GPU e2e using a Float32 intrinsic still passes (the FFI path).
- **No scope creep:** no `Sarek_transpile` (PR-5b), no generator change.

Return GO/NO-GO + findings. NO-GO if: `sarek_stdlib_meta` pulls FFI or fails to populate the
registry FFI-free, OR a golden changed, OR a derived type size differs from the old ctype size, OR
native/interpreter regressed, OR a consumer fails to build (non-pre-existing).
