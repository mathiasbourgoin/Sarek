# Intake + Plan — vulkan-api-split (Intake 7)

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

`sarek-vulkan/Vulkan_api.ml` (1756 lines, no `.mli`) splits along its existing submodule
boundaries (Device, Memory, Stream, Event, Kernel) into one file each, plus a base for
shared helpers. PURE MOVE, byte-identical. The `Vulkan_api.Device.t` / `Memory.buffer` /
`Kernel.args` / etc. access paths used by `Vulkan_plugin_base.ml` must be preserved via
re-export.

## External API (must stay reachable as `Vulkan_api.<path>`)

`Vulkan_plugin_base.ml` uses: `compile_glsl_to_spirv`, `glslang_available`,
`is_available`, and the full `Device.*`, `Memory.*`, `Stream.*`, `Event.*`, `Kernel.*`
submodule surfaces (`.t`, `.buffer`, `.args`, and all their functions).

## Module DAG (verified)

```
base + device  ←  {memory, stream, event}  ←  kernel  ←  Vulkan_api (main, re-exports)
```
Device has no sibling refs. Memory/Stream/Event each reference Device only. Kernel
references Device, Memory, Stream.

## Plan

Each submodule's `struct … end` body becomes the top-level of its own file (so the file
IS the module). All files open the same prelude the original opens: `open Ctypes`,
`open Vulkan_types`, `open Vulkan_bindings`, `open Spoc_framework_registry` (drop any the
compiler flags unused), plus `open Vulkan_api_base` and the sibling-module aliases needed.

1. **`Vulkan_api_base.ml`** (leaf) — lines ~21–167: `memcpy`, `u32`,
   `exception Vk_result_error`, `check`, `compile_glsl_to_spirv_cli`,
   `compile_glsl_to_spirv`, `glslang_available`.
2. **`Vulkan_api_device.ml`** — body of `module Device` (lines ~169–490). `open …_base`.
3. **`Vulkan_api_memory.ml`** — body of `module Memory` (~492–1048). Add
   `module Device = Vulkan_api_device`. `open …_base`.
4. **`Vulkan_api_stream.ml`** — body of `module Stream` (~1050–1136). `module Device = …`.
5. **`Vulkan_api_event.ml`** — body of `module Event` (~1138–1174). `module Device = …`.
6. **`Vulkan_api_kernel.ml`** — body of `module Kernel` (~1176–1737). `module Device = …`,
   `module Memory = Vulkan_api_memory`, `module Stream = Vulkan_api_stream`. `open …_base`.
7. **`Vulkan_api.ml`** (main) — re-export:
   ```ocaml
   exception Vk_result_error = Vulkan_api_base.Vk_result_error
   let compile_glsl_to_spirv = Vulkan_api_base.compile_glsl_to_spirv
   let glslang_available = Vulkan_api_base.glslang_available
   module Device = Vulkan_api_device
   module Memory = Vulkan_api_memory
   module Stream = Vulkan_api_stream
   module Event = Vulkan_api_event
   module Kernel = Vulkan_api_kernel
   let vulkan_version () = …   (* keep verbatim, lines ~1738–1743 *)
   let is_available () = …     (* keep verbatim, lines ~1744–end *)
   ```
   (also re-export `compile_glsl_to_spirv_cli` and `check`/`memcpy`/`u32` only if an
   external caller needs them — the compiler/runtest will tell you; the external list
   above suggests only compile_glsl_to_spirv/glslang_available/is_available + the 5
   submodules.)

## dune

Add `Vulkan_api_base`, `Vulkan_api_device`, `Vulkan_api_memory`, `Vulkan_api_stream`,
`Vulkan_api_event`, `Vulkan_api_kernel` to `(modules …)` in `sarek-vulkan/dune` (before
`Vulkan_api`).

## Quality Gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek-vulkan/
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
ocamlformat --check sarek-vulkan/Vulkan_api*.ml
```
`sarek-vulkan` is `(optional)` — it builds only if Vulkan deps are present. Confirm it
actually compiles in this env (it has FFI bindings; if dune skips it as optional, say so).
Ignore `-lnvrtc`.

## Risks

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| External `Vulkan_api.X.y` path breaks | Med | Med | re-export all 5 submodules + 3 top helpers; build Vulkan_plugin_base |
| Sibling-module alias missing in a file | Med | Low (compile error) | add `module Device/Memory/Stream = …` per DAG |
| Vulkan lib absent → backend not built → split unverified | Med | Med | report whether sarek-vulkan actually compiled; rely on build of the lib |
| Logic changed in move | Low | High | verbatim move |
