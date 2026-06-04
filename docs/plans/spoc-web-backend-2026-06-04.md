# SPOC Web Backend — pure-JS SPOC core + WebGPU backend (design / start)

**Date:** 2026-06-04
**Status:** PROPOSAL — for human review (this is the *start* of the effort; the PR is intentionally not auto-merged)
**Relates to:** the in-browser GPU course program (`in-browser-gpu-course-program-2026-06-04.md`), Phase 5.

## Goal

Split SPOC so its **host-side numeric path compiles to JavaScript** (js_of_ocaml) and add an
in-browser **WebGPU backend plugin** (`Spoc_webgpu`). This lets real SPOC host code run in the
browser — eventually **replacing the hand-written `sarek_webgpu_runner.js`** with the actual SPOC
runtime, and unlocking course lessons on **host-side interaction** (allocating vectors, transfers,
reading back results) and **kernel composition** (chaining kernels, sharing buffers) that the current
canned JS runner cannot teach.

This document is the architecture + phased plan. The accompanying code is only the **first brick**
(a registered-but-inert `Spoc_webgpu` backend skeleton that pins the interface) — deliberately small
so the direction can be reviewed before the invasive decouple lands.

## What the in-browser runner is today (and its ceiling)

`gh-pages/javascripts/sarek_webgpu_runner.js` (Phase 1) is a *standalone* JS reimplementation of "bind
buffers → dispatch → read back", driven by the ABI the transpiler emits. It is perfect for **Level 1/2**
lessons (kernel + structured params, no `Spoc_core` in the browser). It cannot express **host logic**:
multiple allocations with lifetimes, device→device transfers, composing several kernels over shared
buffers, or the real SPOC `Vector`/`Execute` API. Those need actual SPOC running in the browser.

## Boundary map (from the 2026-06-04 source audit — see [[webgpu-runtime-decouple-audit]])

`spoc_core` (`sarek/core/`, links `ctypes` + `unix`) is **mostly Bigarray-based and jsoo-clean**. The
non-jsoo coupling is localized:

- **Custom-type only** (excludable): `Vector_types.ml` `Custom_storage`/`Custom_helpers`,
  `Vector_storage.ml` `Ctypes.allocate_n`/`of_ctypes_ptr`/custom branches, `Vector_transfer.ml`
  `to_ctypes_ptr`. The numeric kinds use `Bigarray_storage` (TypedArray-friendly).
- **Backend signature** (`spoc/framework/Framework_sig.ml`): of the ~59-item `BACKEND`, only
  `Memory.host_ptr_to_device` / `device_to_host_ptr` (l.210–215) are `Ctypes.ptr` — and only custom
  types ever call them. The numeric path uses `host_to_device`/`device_to_host` (`Bigarray.Array1.t`).
- **One core ctypes use:** `Memory.ml:61` `Ctypes_static.sizeof (typ_of_bigarray_kind kind)` →
  replace with a pure `elem_size_of_kind` lookup (4/8/…).
- **unix:** `Profiling.ml` `Unix.gettimeofday` (×2) and `Advanced.ml` `Unix.sleepf` — shim to
  `Sys.time`/jsoo `performance.now`, and a jsoo-friendly yield.

**Verdict:** a ctypes-free numeric `spoc_core` slice is achievable by isolating the custom-type branch
behind a boundary + three small shims. Custom types in-browser are deferred (they `failwith`).

## Backend plugin mechanism (to mirror)

Backends implement `Framework_sig.BACKEND` and self-register on module load via
`Framework_registry.register_backend ~priority (module Backend)` (lazy + `let () = Lazy.force …`).
`Device.init ~frameworks:[…]` looks each up by name. The leanest existing examples to mirror are
`sarek/plugins/interpreter/` (lightest) and `sarek/plugins/native/`. `Spoc_webgpu` will follow the same
shape with `execution_model = JIT` and `generate_source` = the WGSL backend already in `sarek_codegen`.

## Phased plan

1. **(this PR — start) Interface skeleton.** `sarek/plugins/webgpu/` — `Spoc_webgpu` implementing the
   full `Framework_sig.BACKEND` with stub bodies (`failwith "Spoc_webgpu: not yet implemented"`),
   `is_available () = false` (so native `Device.init` never selects it), registered at priority below
   Native. Builds in the normal toolchain. This pins the exact surface the jsoo runtime must fill and
   proves registry integration — **no behaviour yet**.
2. **Core decouple PR.** Isolate the custom-type/ctypes branch behind a module boundary; replace the
   `Ctypes_static.sizeof` use and the two `Unix` uses with pure/shimmable equivalents; add an
   **FFI-free bytecode (and jsoo) build target** for the numeric `spoc_core` slice (the regression gate,
   mirroring the Phase-0B pure-codegen approach). Goldens / native behaviour unchanged.
3. **jsoo WebGPU runtime PR.** Fill `Spoc_webgpu`'s `Memory`/`Kernel`/`Event`/`Stream` with real
   js_of_ocaml ↔ WebGPU bindings (Bigarray ↔ `GPUBuffer`, WGSL `createShaderModule` + compute pass —
   reusing the proven recipe from `sarek_webgpu_runner.js`). Build the in-browser SPOC bundle.
4. **Migrate the playground/runner.** Route the course's "run on your GPU" through the real SPOC web
   runtime instead of `sarek_webgpu_runner.js`; keep the JS runner as a fallback until parity.
5. **Host-interaction & composition courses.** New `learn/` lessons that allocate vectors, transfer,
   run multiple composed kernels over shared buffers, and read back — using the actual SPOC API in the
   browser (impossible with the canned runner).

## Risks / open questions (for the reviewer)

- **WebGPU async vs SPOC's synchronous API.** WebGPU is promise-based; SPOC's `Execute.run_vectors` is
  synchronous. The jsoo runtime must bridge this (Lwt, or a synchronous-looking façade over a
  pre-warmed device). **This is the main design risk** and should be settled before Phase 3.
- **Custom types in-browser:** deferred (stub `failwith`). Acceptable for courses? 
- **Toplevel vs precompiled:** do we need `js_of_ocaml-toplevel` for learners to type host code, or do
  we ship precompiled lesson programs and only let them edit the kernel? (Start with precompiled.)
- **Scope of this first PR:** intentionally just the skeleton — confirm the phasing before the decouple.
</content>
