# Reviewer Sub-brief — pure-codegen-extraction · Phase 0A (spike)

**Status:** VALIDATED

## What was implemented (0A)
A de-risking spike: a golden-snapshot harness, `Backend_error` extracted to a pure lib,
CUDA generator device→framework decouple, a minimal `Float32.sin` registry/stdlib
pure-metadata proof, and an FFI-free bytecode compile of a minimal pure slice.

## Audit first
- The new golden harness: does it capture **byte-exact** output (not fragment asserts), for
  cuda/opencl/metal/glsl, and reset `current_variants`/`current_device` between captures?
  Confirm a deliberate generator tweak fails it (ask the implementer for evidence, or test).
- `Backend_error` extraction: pure lib has no ctypes; the four `*_error` modules repoint
  cleanly; `Framework_sig`/`Device_type`/`Typed_value` stayed in `spoc_framework`.
- CUDA decouple: `current_device:Device.t` → `current_framework:string`; `generate_for_device`
  wrapper preserves SNative output; **CUDA goldens byte-identical**.
- Float32.sin proof: the pure metadata form drops `Spoc_core.Device.t`; the typer still
  resolves the intrinsic; the emitted bytes match the golden.
- FFI-free target: `(libraries …)` contains no `ctypes`/`spoc_core`; it links.

## Risks to verify
- **Determinism:** are the goldens stable across repeated runs (mutable global refs)?
- **Scope creep:** confirm the whole stdlib was NOT converted in 0A (only Float32.sin); and
  native/interpreter paths untouched.
- **Byte-identical:** every backend's goldens unchanged vs the baseline captured in Step 1.

## Expected verdict
GO if: goldens are real byte-exact + deterministic, Backend_error is genuinely ctypes-free,
CUDA + Float32.sin goldens are byte-identical, and the FFI-free target links. Otherwise
NO-GO with the specific blocker (this feeds the 0A go/no-go decision).
