# Intake + Plan — plugin-base-sig

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

CUDA, OpenCL, and Metal plugin-base modules each carry a ~130-line inline module
signature (`module Xxx : sig … end = struct …`) that is structurally identical across
the three. Extract it into a single shared `module type PLUGIN_BASE` in
`spoc/framework/Framework_sig.ml`, and annotate each backend module with it.

**Value:** removes ~260 lines of duplicated signature; one place to evolve the
low-level backend interface.

## Scope correction (audit was partly wrong)

- The audit said "4 backends, shared functor." Reality: **3 backends** (CUDA/OpenCL/Metal).
  Vulkan's `Vulkan_plugin_base.ml` has NO inline signature and a different submodule
  order — explicitly OUT of scope.
- It is a shared **module type**, not a functor. Implementations differ (each wraps a
  distinct FFI api) and are NOT shared.
- No file is oversized (max OpenCL 469 < 500). This is pure DRY, not a size fix.

## Scope Boundary

OUT of scope:
- Vulkan plugin base (no inline sig).
- The implementations (`struct … end` bodies) — unchanged.
- `Framework_sig.BACKEND` — NOT refactored. PLUGIN_BASE is standalone, because
  PLUGIN_BASE.Device has `get_current_device` which BACKEND.Device lacks (they are not
  subset-compatible). Making BACKEND include PLUGIN_BASE is a separate, larger change.
- The `*_plugin.ml` consumers — unchanged (verified they use `Kernel.args` only as an
  abstract variant payload).

## Key design facts (verified by code reading)

- The 3 inline sigs differ in exactly two ways: `is_zero_copy` value ordering (irrelevant
  — OCaml signature matching is order-independent), and CUDA exposes
  `type args = Cuda_api.Kernel.arg list ref` concretely while OpenCL/Metal keep `type args`
  abstract.
- PLUGIN_BASE will declare `type args` **abstract**. Sealing CUDA with it hides the
  concrete `args`. Verified safe: the only consumer, `Cuda_plugin.ml:25`
  (`type Framework_sig.kargs += Cuda_kargs of Cuda_base.Kernel.args`) uses it purely as an
  abstract variant payload, as do OpenCL/Metal equivalents.
- `Framework_sig` already depends on `ctypes` and references `Ctypes.ptr`, so PLUGIN_BASE's
  `host_ptr_to_device` / `device_to_host_ptr` introduce no new dependency.

## PLUGIN_BASE shape

Exactly the current CUDA inline sig (`Cuda_plugin_base.ml:15–145`) with `type args`
abstract. Members: `name`, `version`, `module Device` (with `get_current_device`),
`module Memory` (incl. `alloc_zero_copy`, `is_zero_copy`, ptr transfers, `device_ptr`),
`module Stream`, `module Event`, `module Kernel`, `enable_profiling`,
`disable_profiling`, `is_available`.

## Steps

1. Add `module type PLUGIN_BASE = sig … end` to `Framework_sig.ml` (after `BACKEND` or
   near it). Use abstract `type args`. Internal cross-refs (`Device.t`, `Memory.buffer`,
   `Stream.t`) resolve within the module type. Use unqualified `capabilities`/`dims`
   (we are inside Framework_sig).
2. `sarek-cuda/Cuda_plugin_base.ml`: replace the inline `: sig … end` (lines 15–145) with
   `: Framework_sig.PLUGIN_BASE`. Keep the `struct … end` body verbatim.
3. Same for `sarek-opencl/Opencl_plugin_base.ml` and `sarek-metal/Metal_plugin_base.ml`.
4. Build each backend + framework; build tests; format; license.

## Quality Gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build spoc/framework/ sarek-cuda/ sarek-opencl/ sarek-metal/
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
ocamlformat --check spoc/framework/Framework_sig.ml sarek-cuda/Cuda_plugin_base.ml sarek-opencl/Opencl_plugin_base.ml sarek-metal/Metal_plugin_base.ml
./scripts/check-license-headers.sh
```

## Risks

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| A backend struct lacks a PLUGIN_BASE member → seal fails | Low | Low (compile error) | normalized diff showed identical member sets; build is the oracle |
| Abstracting CUDA `args` breaks a consumer | Low | Med | verified only abstract-payload use; build gates |
| `get_current_device` missing in some backend struct | Low | Low (compile error) | present in all 3 inline sigs per diff |
