# QA Scope — pure-codegen-extraction · Phase 0A (spike)

**Status:** VALIDATED

## Gates (all must pass)
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed .ml files>
# FFI-free proof target builds (no ctypes/spoc_core)
```
(`-lnvrtc` link error in a full `dune build` is a pre-existing CUDA-absent issue. The CI
`build` job's `e2e-fast` matrix-mul segfault is a known flaky OpenCL-CPU failure — re-run
to confirm, don't treat a lone occurrence as a regression.)

## Behaviours to validate
- **Goldens:** the new snapshot test passes AND is byte-exact (introduce a temp 1-char
  change to a generator → it must fail; revert). Run the suite twice → identical (determinism).
- **No regression:** existing `test_sarek_ir_{cuda,opencl,metal,glsl}` fragment tests still pass.
- **GPU runtime sanity (hardware available — RX 7900 XTX):** the Vulkan e2e still passes,
  proving the GLSL `?log` decouple and any glsl touch didn't change runtime behaviour:
  `dune exec sarek/tests/e2e/test_vector_add.exe -- --vulkan`
  (and `test_klet_variant --vulkan` for variant codegen).
- **FFI-free proof:** the Step-5 target compiles without ctypes/spoc_core in its libraries.

## Out of scope for 0A QA
- `Sarek_transpile.of_source` (0B).
- jsoo bundle (Phase 2).
- Full-stdlib conversion (only Float32.sin in 0A).
