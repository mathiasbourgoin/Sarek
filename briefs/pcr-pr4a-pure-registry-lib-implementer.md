# Implementer Sub-brief — pure-codegen-rollout PR-4a (make sarek_registry pure)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-4 of 5, part a)
**Type:** refactor (decouple; pure — NO behavior/codegen change)

## Goal
Make the `sarek_registry` library (`spoc/registry/Sarek_registry.ml`, 260 lines, deps
`spoc_framework`) **ctypes/FFI-free** by retyping its device-codegen closures from
`Spoc_framework.Device_type.t -> string` to `string -> string` (a framework-name string),
mirroring the proven Phase 0B `Device.t -> framework` decouple. This unblocks PR-4b (extracting
the 4 generators — which call `Sarek_registry.fun_device_template` — into a pure `sarek_codegen`).

**Pure refactor: generated source must be BYTE-IDENTICAL.** `sarek/tests/codegen_golden/` is the
oracle. Any golden diff = regression → STOP.

## User decision (do NOT re-litigate)
Chosen approach: make `sarek_registry` pure by retyping its `*_device` closures. The line-384
generator query (`Sarek_registry.fun_device_template`) stays as-is — one registry mechanism,
just decoupled from `Device_type.t`. (NOT unifying onto `Sarek_pure_registry`.)

## Exact changes in `spoc/registry/Sarek_registry.ml`
1. **`ti_device`** (l.45) `Spoc_framework.Device_type.t -> string` → `string -> string`.
2. **`fi_device`** (l.73) same retype.
3. **`type_device_code`** (l.152-155) `ti.ti_device dev` — param `dev` is now a framework string.
4. **`fun_device_code`** (l.158-163) `fi.fi_device dev` — param now a framework string.
5. **`fun_device_template`** (l.167-197): DELETE the `minimal_dev` `Device_type.t` record
   (l.172-194) and call `Some (fi.fi_device "generic")`. RATIONALE: the old code passed
   `minimal_dev` with `framework = "generic"`, and `cuda_or_opencl` maps "generic" → the `_`
   (CUDA) branch. Passing the string `"generic"` reproduces this EXACTLY → byte-identical output.
6. **`cuda_or_opencl`** (l.252-256): change `(dev : Spoc_framework.Device_type.t)` →
   `(framework : string)` and match `framework` directly (drop the `dev.framework` projection).
   Keep the exact arms: `"OpenCL" -> opencl_code | "CUDA" | "Native" | "Interpreter" | _ -> cuda_code`.
7. **`register_type`/`register_fun` `~device` closures** (e.g. l.245-246 `fun _ -> "int"`):
   these ignore their arg — they typecheck unchanged under `string -> string`. Verify, don't rewrite.
8. Remove `spoc_framework` from `spoc/registry/dune` `(libraries …)` IF the build confirms it is
   then unused. If something else still needs it, STOP and report (the grep says only
   `Device_type.t` used it).

## Stdlib callers to update (5 files — mechanical signature change)
Each has `let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl` where `d` was a
`Device_type.t`; now `d` is the framework string. The body is unchanged (it just forwards `d`).
Confirm each still typechecks and that the `*_device` closures these build are now `string -> string`:
- `sarek/Sarek_stdlib/Float32.ml:39`
- `sarek/Sarek_float64/Float64.ml:30`
- `sarek/Sarek_stdlib/Int32.ml:29`
- `sarek/Sarek_stdlib/Int64.ml:29`
- `sarek/Sarek_stdlib/Math.ml:13`

## Tests to update
`spoc/registry/test/test_sarek_registry.ml` — `test_cuda_or_opencl` (l.286-334) and
`test_fun_device_template` (l.270-282) construct `Device_type.t` devices and call the helpers.
Update them to pass framework strings (`"CUDA"`, `"OpenCL"`, `"Native"`). Preserve the asserted
outcomes (l.331-333: CUDA→cuda_code, OpenCL→opencl_code, Native→cuda_code).

## Hard constraints
- **BYTE-IDENTICAL goldens** — `@sarek/tests/runtest` unchanged. The generators' line-384 query
  output must not change. A golden diff is a regression → STOP.
- `sarek_registry`'s `(libraries …)` must end up WITHOUT `spoc_framework` (the PR's purpose).
- Native/interpreter behaviour identical — run their intrinsic tests.
- NO generator changes (PR-4b), no `sarek_codegen`, no `Sarek_transpile`.
- SPDX headers preserved; `dune fmt` clean.
- **COMMIT after each logical step on `phase0b/pr4a-pure-registry-lib`** (do not leave uncommitted).

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build @spoc/registry/runtest # registry unit tests
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune exec sarek/tests/e2e/test_math_intrinsics.exe -- --interpreter
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed .ml/dune>
```
(`-lnvrtc` full-build link error pre-existing; CI e2e-fast matrix-mul segfault known-flaky.)
