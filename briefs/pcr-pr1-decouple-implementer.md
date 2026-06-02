# Implementer Sub-brief — pure-codegen-rollout PR-1 (opencl/metal/glsl decouple)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-1 of 5)

## Goal
Mirror the merged Phase 0A CUDA decouple onto the OpenCL and Metal generators, and decouple
GLSL from `Spoc_core.Log`. Pure refactor — **byte-identical** generated source (the
`sarek/tests/codegen_golden/` harness on `main` is the oracle). This removes the
`Spoc_core.Device.t`/`Spoc_core.Log` couplings from three more generators; it does NOT yet
extract them into a pure lib (PR-4) or touch the registry (PR-2).

## Reference (already merged, reviewed GO)
`sarek-cuda/Sarek_ir_cuda.ml` on `main`: `current_device : Device.t option ref` →
`current_framework : string option ref`; `SNative` branch reads the framework string;
`generate_for_device ~device` sets `current_framework := Some device.Spoc_core.Device.framework`;
`open Spoc_core` removed. Replicate exactly.

## Steps (build + goldens green after each)
1. **OpenCL** `sarek-opencl/Sarek_ir_opencl.ml`: replace `current_device : Device.t option ref`
   (l.~35) with `current_framework : string option ref`; the `SNative` branch (l.~531) reads
   the framework string; `generate_for_device` (l.~770) becomes the wrapper; drop `open
   Spoc_core` if now unused (qualify any residual refs).
2. **Metal** `sarek-metal/Sarek_ir_metal.ml`: same (l.~24 / ~644 / ~997).
3. **GLSL** `sarek-vulkan/Sarek_ir_glsl.ml`: replace the two `Spoc_core.Log.debugf` calls
   (l.~947, ~1034) with an injected `?log:(string -> unit)` parameter defaulting to no-op on
   the relevant generate entry; the Vulkan backend wrapper passes
   `~log:(fun s -> Spoc_core.Log.debugf Spoc_core.Log.Device "%s" s)`. Remove glsl's
   `Spoc_core` coupling if these were its only uses.

## Hard constraints
- BYTE-IDENTICAL: the opencl/metal/glsl goldens in `sarek/tests/codegen_golden/` must not
  change. If a golden would change, STOP — that's a real output regression, not expected.
- Do NOT touch CUDA (done), the pure registry, or the stdlib (PR-2). No lib split (PR-3/4).
- SPDX headers preserved; `dune fmt` clean.

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed .ml>
# GPU sanity (RX 7900 XTX): dune exec sarek/tests/e2e/test_vector_add.exe -- --vulkan
```
(`-lnvrtc` full-build link error is pre-existing; CI e2e-fast matrix-mul segfault is known-flaky.)

## Plan note
Low risk — proven mirror of the 0A CUDA change (reviewed GO, merged). Dual-voice not
re-spawned for this PR per that precedent. If OpenCL/Metal/GLSL diverge from the CUDA
pattern in some way that changes output, escalate rather than force.
