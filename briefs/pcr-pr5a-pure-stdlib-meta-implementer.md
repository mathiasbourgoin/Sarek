# Implementer Sub-brief — pure-codegen-rollout PR-5a (pure stdlib metadata lib)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-5 of 5, part a)
**Type:** refactor (structural; pure — NO behavior/codegen change)

## Goal
Make the stdlib intrinsic **metadata** (the registrations the typer + codegen read to resolve
`Float32.sin`, `Int32.add`, `Gpu.*`, etc.) populate **FFI-free**, so a transpile (PR-5b) can
resolve stdlib intrinsics without linking the ctypes/spoc_core `sarek_stdlib`. Split the stdlib
registration into a PURE metadata lib + the existing FFI lib (ctype/Vector/native execution).

**Pure refactor: generated source must be BYTE-IDENTICAL** (`sarek/tests/codegen_golden/`), AND
native/interpreter behaviour identical. Any golden diff = STOP.

## User decision (do NOT re-litigate)
Chosen "Full": stdlib intrinsics must transpile FFI-free (completes the registry-unification
deferred since PR-2). MVP-only (basic kernels) was explicitly rejected.

## Evidence (already verified — do NOT re-investigate)
- `%sarek_intrinsic` (`sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`) emits, per intrinsic:
  - TYPE: `Sarek_registry.register_type ~device ~size:(Ctypes.sizeof ctype)` (l.223-226) +
    `Sarek_ppx_registry.register_type (make_type_info ~size:(Ctypes.sizeof ctype) ~sarek_type)` (l.230-234).
  - FUN: `Sarek_registry.register_fun ~device ~arg_types ~ret_type` (l.352) +
    `Sarek_ppx_registry.register_intrinsic (make_intrinsic_info …)` (l.364). **No ctype/size — already pure.**
- The ONLY ctypes tie in registration is `Ctypes.sizeof ctype` in the TYPE path. `register_type ~size`
  takes a plain `int` (`Sarek_registry.ml:92`), so a literal/derived int works with no Ctypes.
- The `ctype = Ctypes.float` field value (e.g. `Float32.ml:31`) and the native execution closures
  (`Spoc_core.Vector`) are the remaining FFI; they belong to the FFI runtime path only.
- `Sarek_env.with_stdlib` reads `Sarek_ppx_registry.all_intrinsics ()` (populated at module-init by
  whichever lib ran the registrations). Core primitives come from the pure `Sarek_core_primitives`.

## Recommended mechanism (escalate if it balloons — see clause)
1. **PPX size without Ctypes.** Make `Sarek_ppx_intrinsic` derive the type `~size` from the
   `sarek_type` (TFloat32→4, TInt32→4, TFloat64→8, TInt64→8, TBool→4, TChar→1, Custom→provided) via
   a pure helper, INSTEAD of `Ctypes.sizeof ctype`. Make the `ctype` field OPTIONAL in the
   `%sarek_intrinsic` type record so a pure-meta module can omit it. The generated registration code
   then references no Ctypes. (FUN path already pure.) Build + goldens green. COMMIT.
2. **Create `sarek_stdlib_meta`** (pure lib, deps `ppxlib sarek_registry sarek_frontend` +
   `(pps sarek_ppx_intrinsic)`, NO `spoc_core`/`ctypes`): holds the metadata definitions for
   Float32/Float64/Int32/Int64/Math/Gpu — the `%sarek_intrinsic` declarations WITHOUT `ctype` and
   WITHOUT native Vector closures. Its module-init registers name/sarek-type/device-template/size
   into `Sarek_ppx_registry` + `Sarek_registry`. Build + goldens green. COMMIT.
3. **Reduce `sarek_stdlib` (FFI)** to depend on `sarek_stdlib_meta` and add ONLY the ctype +
   native/Vector execution layer (the `ctype` marshalling + native fallbacks). It must NOT
   re-register the metadata (avoid double-registration — `register_*` is `Hashtbl.replace`, so a
   duplicate identical registration is harmless, but prefer single-source). Native/interpreter
   behaviour identical. Build + goldens green. COMMIT.
4. **Prove FFI-free population:** add a small test/exe that links `sarek_frontend` +
   `sarek_stdlib_meta` (NO spoc_core/ctypes) and asserts `Sarek_ppx_registry.all_intrinsics ()`
   contains `Float32.sin` (and a few others) with the right type — i.e. the typer could resolve it.
   This is the PR-5a proof that PR-5b builds on. COMMIT.
5. **Final gates.** COMMIT format.

### Escalation clause (mirrors PR-2)
If making the `%sarek_intrinsic` size-derivation pure + ctype-optional proves too large/risky in a
bounded attempt, STOP and report with what you found — do NOT force a half-migration or break the
FFI stdlib. A documented partial (e.g. only Float32/Float64/Int32/Int64/Math, deferring Gpu) is
acceptable IF you say exactly what's covered and the FFI-free proof passes for the covered set.

## Hard constraints
- **BYTE-IDENTICAL goldens** (`@sarek/tests/runtest`; `git diff origin/main -- sarek/tests/codegen_golden/` empty).
- Native/interpreter intrinsic behaviour identical — `test_math_intrinsics --native` and `--interpreter` pass.
- `sarek_stdlib_meta` `(libraries …)` MUST have NO `spoc_core`/`ctypes`. The FFI-free proof exe links no FFI.
- The existing FFI `sarek_stdlib` consumers (plugins, e2e, native exec) keep working unchanged.
- SPDX headers; `dune fmt` clean (COMMIT fmt). **COMMIT after each step on `phase0b/pr5a-pure-stdlib-meta`.**

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek_stdlib_meta       # pure, no ctypes/spoc_core
opam exec --switch=/home/mathias/dev/SPOC -- dune build                         # all consumers
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune exec sarek/tests/e2e/test_math_intrinsics.exe -- --native
opam exec --switch=/home/mathias/dev/SPOC -- dune exec sarek/tests/e2e/test_math_intrinsics.exe -- --interpreter
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed>
```
(`-lnvrtc` full-build link error pre-existing; CI e2e-fast matrix-mul segfault known-flaky.)
