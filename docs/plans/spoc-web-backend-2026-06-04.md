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

## Risk mitigation

### Risk 1 — WebGPU async vs SPOC's synchronous API  →  **mitigated (measured)**

The risk is narrower than "async vs sync". In WebGPU **almost everything is synchronous** —
`createBuffer`/`writeBuffer`/`createShaderModule`/`createComputePipeline`/`createBindGroup`/the
encoder/`dispatchWorkgroups`/`queue.submit` are all sync. Only **two** ops are promises:
**device acquisition** (`requestAdapter`/`requestDevice`, one-time) and **read-back** (`buffer.mapAsync`).

Mitigations, cheapest → most thorough:

| | Approach | Cost |
|---|---|---|
| **A** | **Pre-warm the device**: acquire adapter+device once at init; the harness awaits it before enabling "Run", so in-browser `Device.init` returns the handle synchronously. Removes async (1). | trivial |
| **B1** *(chosen)* | **Async read-back seam**: `Execute.run_vectors` stays synchronous (all sync WebGPU calls); only read-back is async in the browser — `Vector.to_array_async : 'a vector -> 'a array Lwt.t` (native keeps sync `to_array`). The one async point a lesson hits is "read results, then check". | small |
| **C** | **Fully-synchronous façade**: jsoo in a Web Worker, GPU service on the main thread, sync RPC over `SharedArrayBuffer` + `Atomics.wait`, COOP/COEP via a coi-service-worker on GitHub Pages. SPOC's host API stays byte-identical/synchronous (needed only for a future toplevel). | heavy |

Scoping the courses to **precompiled lesson programs** (the learner edits the kernel, as today) means the
async seam is written once by the lesson author, so **A + B1 fully suffice and C is unneeded now.**

### Risks 2–5 — mitigation

- **Custom types (`Ctypes.ptr`):** out of v1 scope (stub `failwith`; courses use numeric vectors). Later,
  back `Custom_storage` with an `ArrayBuffer`/`DataView` (no ctypes); `Custom_helpers.read_float32`/
  `write_int64` map 1:1 to `DataView.getFloat32`/`setBigInt64`. The byte-identical golden gate protects
  the numeric path during the decouple.
- **`Ctypes_static.sizeof` + `Unix`:** fully removable — pure `elem_size_of_kind` lookup (proven by the
  spike) and an injectable clock ref (native default / jsoo `performance.now`); the futures `Unix.sleepf`
  is excluded from the numeric build.
- **jsoo won't no-op ctypes-foreign:** don't compile the FFI plugins to jsoo; route through the
  `Spoc_webgpu` BACKEND. Add a `dune` bytecode/jsoo target that builds only the numeric slice and **fails
  the build if ctypes re-enters the link** (Phase-0B FFI-free gate) — converts the latent risk into CI.
- **Toplevel vs precompiled:** start **precompiled** (avoids `js_of_ocaml-toplevel`, neutralizes most of Risk 1).

## De-risking spikes — RESULTS (both passed; on this branch under `sarek/plugins/webgpu/spike/`)

1. **jsoo ↔ WebGPU binding** (`spike/webgpu_binding/`) — an OCaml program compiled with js_of_ocaml drives a
   full WebGPU `vector_add` **from OCaml** (`Js.Unsafe` bindings: `requestAdapter`→`requestDevice`→buffers→
   WGSL `createShaderModule`→pipeline→dispatch→`mapAsync`), returning the result to OCaml via the **B1 async
   callback seam**. **Verified correct for all 256 elements on the real RX 7900 XTX** (Playwright + flagged
   Chrome). ⟹ Risk 1's chosen path is proven end-to-end.
2. **FFI-free numeric core** (`spike/numeric_core/`) — a Bigarray-backed numeric vector (`create/get/set/
   to_array`, `elem_size_of_kind`, injectable clock) **builds to both bytecode and js_of_ocaml with deps =
   `bigarray` only** (no ctypes/unix); the bytecode test PASSes. The gap analysis classified every
   `Ctypes`/`Unix` occurrence in the seven `spoc_core` numeric-path files and returned **GO**: the scalar
   path is FFI-free once (i) `Memory.ml:61 sizeof` is shimmed, (ii) the custom-type branch is gated behind
   a boundary, (iii) `Unix` is shimmed. One refinement over the original audit: the **FFI-transfer pointer
   plumbing** (`Memory.ml` 130/141/172, `Vector_transfer.ml` 15–36 — `bigarray_start`/`to_voidp`) exists only
   to feed the native FFI backends and must also be isolated for a *link-clean* build (it's never reached on
   the WebGPU/typed-array path).

## Remaining decisions (for the reviewer)

- Confirm the **5-step phasing** and the **B1 (async read-back seam) + precompiled** direction before the decouple.
- The decouple's exact module boundary: a `spoc_core_ctypes` companion lib vs a `Custom` functor for the
  custom-type + FFI-transfer surface — to be settled at the start of the Core-decouple PR.
- **Scope of this first PR:** still just the skeleton + these two spikes — no behaviour/decouple yet.
</content>
