# Vulkan Backend

## Component Inventory

- `sarek-vulkan/README.md`: package documentation and examples.
- `sarek-vulkan/dune`: optional `sarek-vulkan.plugin` library; module list and toolchain comments are at `sarek-vulkan/dune:11-33`.
- `sarek-vulkan/Vulkan_error.ml`: shared backend error helpers.
- `sarek-vulkan/Vulkan_types.ml`: Vulkan handles, constants, result codes, structs, and helper constructors.
- `sarek-vulkan/Vulkan_bindings.ml`: dynamic Vulkan loader and raw FFI declarations.
- `sarek-vulkan/Shaderc.ml`: optional shaderc FFI compiler path.
- `sarek-vulkan/Vulkan_api.ml`: reduced to re-exports of the per-submodule files plus `vulkan_version` and `is_available` (`sarek-vulkan/Vulkan_api.ml:31-39`).
- `sarek-vulkan/Vulkan_api_base.ml`: `memcpy`, `u32`, `check`, GLSL-to-SPIR-V compilation (shaderc and `glslangValidator` CLI), and `glslang_available`.
- `sarek-vulkan/Vulkan_api_device.ml`: device enumeration, capabilities, device cache, and destroy.
- `sarek-vulkan/Vulkan_api_memory.ml`: buffer allocation, staging buffers, host/device copies, and `device_to_device`.
- `sarek-vulkan/Vulkan_api_stream.ml`: command pool/buffer streams and default-stream cache.
- `sarek-vulkan/Vulkan_api_event.ml`: fence-based events.
- `sarek-vulkan/Vulkan_api_kernel.ml`: SPIR-V module/descriptor layout/pipeline creation, kernel cache, argument binding, push constants, and compute dispatch (`launch`).
- `sarek-vulkan/Vulkan_plugin_base.ml`: framework backend implementation.
- `sarek-vulkan/Vulkan_plugin.ml`: registration, intrinsic registry, source generation, and external GLSL execution.
- `sarek-vulkan/Sarek_ir_glsl.ml`: Sarek IR to GLSL compute shader generator.
- `sarek-vulkan/test/`: `test_vulkan_error.ml`, `test_sarek_ir_glsl.ml`, and test `dune`.

## Per-File Purpose

- `Vulkan_error.ml` instantiates `Backend_error.Make`.
- `Vulkan_types.ml` defines Vulkan constants and ctypes structures. `vk_physical_device_properties` is modeled only through `deviceName` plus a 1024-int padding array at `sarek-vulkan/Vulkan_types.ml:399-428`.
- `Vulkan_bindings.ml` lazily loads `libvulkan` at `sarek-vulkan/Vulkan_bindings.ml:21-50` and binds instance/device/memory/buffer/shader/pipeline/descriptor/command/fence APIs.
- `Shaderc.ml` lazily loads `libshaderc` at `sarek-vulkan/Shaderc.ml:17-45` and compiles GLSL to SPIR-V with a process-wide compiler ref at `sarek-vulkan/Shaderc.ml:159-167`.
- `Vulkan_api.ml` is now a thin facade: it re-exports the `Vulkan_api_*` submodules (base/device/memory/stream/event/kernel) and defines only `vulkan_version` and `is_available` (`sarek-vulkan/Vulkan_api.ml:11-39`). The behavior below is unchanged; it was moved verbatim into the submodule files.
- `Vulkan_api_base.ml` compiles GLSL via shaderc or `glslangValidator` and provides shared `memcpy`/`check` helpers. `Vulkan_api_device.ml` creates Vulkan devices and queries capabilities. `Vulkan_api_memory.ml` allocates buffers and handles staging transfers. `Vulkan_api_kernel.ml` builds descriptor layouts/pipelines, manages command buffers/fences, and dispatches compute shaders. `Vulkan_api_stream.ml`/`Vulkan_api_event.ml` provide streams and fence-based events.
- `Vulkan_plugin_base.ml` exposes framework devices, memory, stream/event, kernel args, and launch over `Vulkan_api`.
- `Vulkan_plugin.ml` generates GLSL, supports external GLSL source, and registers with priority 80.
- `Sarek_ir_glsl.ml` emits GLSL compute shaders with descriptor-set storage buffers and push constants for scalar and vector length parameters. Variant emission and `mangle_name` now delegate to the shared `Sarek_ir_codegen` module: `mangle_name` is aliased (`sarek-vulkan/Sarek_ir_glsl.ml:32`) and `gen_variant_def` calls `Sarek_ir_codegen.gen_variant_def_glsl` (`sarek-vulkan/Sarek_ir_glsl.ml:968-969`).
- `Vulkan_plugin_base.ml`'s `Vulkan` module is NOT constrained to `Framework_sig.PLUGIN_BASE` (it never carried an inline signature, so it was not part of the shared-signature dedup) (`sarek-vulkan/Vulkan_plugin_base.ml:16`).

## Features and APIs

- Vulkan compute backend with GLSL-to-SPIR-V through shaderc or `glslangValidator`.
- Device enumeration, memory allocation with host-visible/device-local selection, staging copies for non-mappable buffers, command pool/buffer/fence streams, descriptor-set based buffer binding, push constants, and compute dispatch.
- External source language support is GLSL only at `sarek-vulkan/Vulkan_plugin.ml:207`.
- Generated GLSL uses `layout(local_size_x=...)`, storage buffers for vectors, push constants for scalar params and vector lengths, and Vulkan built-in IDs.

## Invariants

- Vulkan ctypes structures must match ABI layout exactly.
- Cache keys must include every semantic input to compilation and pipeline creation.
- Descriptor set layout bindings must match shader-declared binding numbers.
- Push constant layout and runtime byte writes must match generated GLSL layout.
- Command buffers must be in the correct reset/record/submit state.
- Staging memory mapped by `vkMapMemory` must be unmapped before freeing.

## Potential Invariant Violations and Bugs

- `vk_physical_device_properties` is represented by partial fields plus padding at `sarek-vulkan/Vulkan_types.ml:399-428`. If the padding/layout is wrong for the Vulkan headers/platform ABI, `vkGetPhysicalDeviceProperties` can corrupt memory. Marked high-risk ABI issue.
- CLI GLSL compilation builds shell command strings with temp filenames at `sarek-vulkan/Vulkan_api_base.ml:44-56` and debug copy commands at `sarek-vulkan/Vulkan_api_base.ml:87-93`; paths are not shell-escaped. Temp paths are usually safe, but this should use process argv or proper quoting.
- **FIXED (verified 2026-07-02):** `compile_glsl_to_spirv_cli` previously accepted `entry_point` but ignored it in the CLI command. It now passes `-e <entry_point>` to `glslangValidator` (`sarek-vulkan/Vulkan_api_base.ml:54-56`).
- **FIXED (verified 2026-07-02):** `Device.destroy` previously left cached device/default-stream state after destroying Vulkan handles. It now removes the cache entry: `Hashtbl.remove device_cache dev.id` runs before the Vulkan teardown calls in `Device.destroy` (`sarek-vulkan/Vulkan_api_device.ml:311-312`).
- **Softened (verified 2026-07-02):** the staging-buffer cleanup closures (`sarek-vulkan/Vulkan_api_memory.ml:449-451`, `:482-484`, `:516-518`, `:550-552`) call `vkDestroyBuffer`/`vkFreeMemory` without an explicit `vkUnmapMemory` call, and each is wrapped in `Fun.protect ~finally:free_staging`. Per the Vulkan spec, `vkFreeMemory` implicitly unmaps any current mapping, and the cleanup itself is guaranteed to run via `Fun.protect`. The KB previously overclaimed this as a bug; there is no unmap-before-free hazard here. (`Vulkan_api_memory.ml:free` for user-owned buffers, line ~419-425, does call `vkUnmapMemory` explicitly when `mapped_ptr` is set, which remains correct.)
- Allocation computes `size * elem_size` without validation at `sarek-vulkan/Vulkan_api_memory.ml:190-192` and `sarek-vulkan/Vulkan_api_memory.ml:306-313`.
- Copy APIs do not validate requested byte counts against source and destination capacities at `sarek-vulkan/Vulkan_api_memory.ml:409-543`.
- `device_to_device` is not implemented at `sarek-vulkan/Vulkan_api_memory.ml:545-547`, even though Vulkan buffer copy commands are already used for staging transfers.
- **Re-verified 2026-07-02, still live:** buffer binding count is inferred by regex-counting occurrences of `binding = N` (`Str.regexp "binding *= *[0-9]+"`) at `sarek-vulkan/Vulkan_api_kernel.ml:197-207` (line numbers shifted from the prior citation `:127-143` due to added code; re-confirmed present), then descriptor layout bindings are created as dense `0..num_bindings-1`. Sparse binding numbers or comments can still produce invalid layouts. Do not close — this exact pattern (`Str.regexp`) was checked against current source and is unchanged.
- **Re-verified 2026-07-02, still live:** pipeline creation hardcodes entry point `"main"` (`sarek-vulkan/Vulkan_api_kernel.ml:303`, `setf stage_info shader_stage_pName "main"`), ignoring the requested kernel name. This is the compiled-shader entry point, distinct from the fixed CLI `-e` flag (the CLI fix controls what `glslangValidator` compiles *from*, not what the pipeline stage records as its entry point). Do not close.
- **fixed 2026-07-02 (merged):** `compile_cached` previously omitted the kernel name from its in-memory key. It now builds the key via the shared `Spoc_framework.Compile_cache.make_key ~device ~name ~source ()` (`Vulkan_api_kernel.ml:385-390`), matching the CUDA fix.
- **fixed 2026-07-02 (merged):** `set_arg_buffer` previously ignored the supplied index and assigned sequential bindings. Buffer args are now stored in a dedicated `Spoc_framework.Kernel_args.t` (`buffer_store`, `Vulkan_api_kernel.ml:55`), and Vulkan descriptor binding numbers are derived deterministically from the caller-supplied idx: `resolve_bindings` (`Vulkan_api_kernel.ml:63-75`) ranks buffers by ascending idx and assigns binding N to the Nth-smallest — "order-of-call-independent, unlike the previous per-call sequential counter it replaces" (code comment). A companion `validate_buffer_indices` (`Vulkan_api_kernel.ml:90-114`) rejects negative indices and index-count mismatches (catching a caller that under/over-supplies buffer args) before `resolve_bindings` can silently compress a bad index set into a valid-looking dense range.
- **fixed 2026-07-02 (merged):** push-constant setters (`set_arg_int32`, `set_arg_int64`, `set_arg_float32`, `set_arg_float64`) now bounds-check the 128-byte push-constant block before writing and raise a structured `Vulkan_error` ("push constant block overflow") on overflow, instead of writing past the buffer (`sarek-vulkan/Vulkan_api_kernel.ml:501-510`, `write_at`/`push_constant_limit`).
- **fixed 2026-07-02 (merged) — logical-index binding + 8-byte alignment (this was the "generated GLSL declares vector-length push constants but runtime binding only records buffers" bug flagged previously):** scalar push-constant arguments are now stored in a separate `scalar_store : scalar_arg Spoc_framework.Kernel_args.t` (`Vulkan_api_kernel.ml:59`), keyed by the caller's own idx, decoupled from buffer indices. At launch, `build_push_constants` (`Vulkan_api_kernel.ml:465-520`) assembles the push-constant byte block by (1) vector lengths, in ascending-buffer-idx order (matching `resolve_bindings`'s buffer ordering, which is what `Sarek_ir_glsl.gen_push_constants` assumes), then (2) user scalars, in ascending scalar-idx order — reproducing the exact grouping the GLSL codegen emits (`sarek/codegen/Sarek_ir_glsl.ml:889-919`), rather than relying on argument-setting call order. It also reproduces GLSL's std430-like base-alignment: `align_to width` inserts padding before any 8-byte field (`int64_t`/`double`) following a 4-byte field, so mixed 32/64-bit push-constant blocks no longer read shifted bytes after the first misalignment (`Vulkan_api_kernel.ml:491-500`). Buffer-idx negative values are rejected the same way as `validate_buffer_indices` above.
- `launch` ignores the runtime block size at `sarek-vulkan/Vulkan_api_kernel.ml:394-397`; block size is baked into generated GLSL, but direct external source launch can mismatch user expectations.
- **FIXED (verified 2026-07-02):** command-buffer reset/reuse discipline. The command pool is now created with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` (`sarek-vulkan/Vulkan_api_stream.ml:27`), and before re-recording, the kernel launch path now waits on the fence and resets it (`vkWaitForFences` then `vkResetFences`) before `vkBeginCommandBuffer` (`sarek-vulkan/Vulkan_api_kernel.ml:478-492`). The previously-flagged "begin without explicit reset" state bug no longer applies.
- Generated GLSL maps `TInt64` and `TFloat64` to `int64_t` and `double` at `sarek-vulkan/Sarek_ir_glsl.ml:147-149`, but the emitted header at `sarek-vulkan/Sarek_ir_glsl.ml:784-795` does not enable int64/fp64 extensions. README examples show extensions that implementation does not emit.
- `SNative` is emitted as a comment instead of executable native GPU code at `sarek-vulkan/Sarek_ir_glsl.ml:682-685`.
- **New entry, verified 2026-07-02 — do not close:** `generate_source` catches all exceptions and returns `None` at `sarek-vulkan/Vulkan_plugin.ml:194` (`with _ -> None`), hiding codegen failure reasons, matching the same pattern already tracked for OpenCL/Metal. Confirmed still present against current source; not touched by #213/#214.

## Performance and Maintainability Risks

- `memcpy` is looked up inside the helper function on each call at `sarek-vulkan/Vulkan_api_base.ml:10-27`.
- Shaderc uses a global compiler ref without release or locking at `sarek-vulkan/Shaderc.ml:159-167`.
- Descriptor pool sizing is fixed at ten sets and `num_bindings * 10` descriptors at `sarek-vulkan/Vulkan_api_kernel.ml:260-277`.
- Device creation requests Vulkan 1.2 in instance app info at `sarek-vulkan/Vulkan_api_device.ml:72-82`; older drivers that could support the used subset may be rejected.
- README and implementation drift: README generated shader snippets include extensions not emitted, and API examples reference names such as `Memory.malloc` that do not match the current code.

## Related Tests

- `sarek-vulkan/test/dune:3-9` defines two Alcotest executables.
- `sarek-vulkan/test/test_vulkan_error.ml:143-162` covers shared error constructors and formatting.
- `sarek-vulkan/test/test_sarek_ir_glsl.ml:219-243` covers basic GLSL codegen fragments.

## Missing Tests

- SPIR-V compile tests for generated shaders with buffers, push constants, records, variants, int64, and fp64.
- Descriptor binding extraction tests with sparse or out-of-order binding numbers.
- Runtime arg binding tests for vector lengths and scalar push constants.
- Command buffer reuse test across repeated launches.
- Staging transfer cleanup test or validation-layer run to catch mapped-memory/free mistakes.
- Device destroy/cache invalidation test.
- External GLSL source test where entry point is not `main`.

## Concrete Improvement Candidates

- Replace regex binding inference with structured binding metadata from codegen or SPIR-V reflection. **Still open, re-verified 2026-07-02** — regex inference (`Str.regexp "binding *= *[0-9]+"`) unchanged.
- Include `name` in cache keys and use the requested entry point consistently across shaderc, CLI compilation, and pipeline stage creation. **Updated 2026-07-02: cache key now DONE** — `compile_cached` uses `Compile_cache.make_key ~name ...` (`Vulkan_api_kernel.ml:385-390`). CLI compilation already honored `-e entry_point`. Pipeline-stage entry point is **still hardcoded to `"main"`** (`Vulkan_api_kernel.ml:303`) — that part remains open.
- ~~Reset command buffers before re-recording or allocate one-time command buffers per launch.~~ DONE 2026-07-02 — pool created with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` and fence wait/reset added before `vkBeginCommandBuffer`.
- Emit required GLSL extensions for int64/fp64 based on IR usage and device support. Still open.
- Implement `device_to_device` with `vkCmdCopyBuffer` and size validation. Still open — kept live per the copy-size-validation-gap instruction.
- ~~Add bounds checks for push constant writes and derive push constant layout from the same metadata as codegen.~~ **DONE 2026-07-02 (merged)** — bounds checks raise on 128-byte overflow (`Vulkan_api_kernel.ml:501-510`); push constants are now also bound by logical index rather than call order, and 64-bit fields are correctly 8-byte aligned (`build_push_constants`, `Vulkan_api_kernel.ml:465-520`). Deriving the *binding-number* layout from shared codegen metadata (as opposed to the regex-based `num_bindings` count) is still outstanding.
- ~~Bind buffer arguments by logical index instead of sequential call order.~~ **DONE 2026-07-02 (merged)** — `resolve_bindings` ranks by ascending caller idx (`Vulkan_api_kernel.ml:63-75`); `validate_buffer_indices` rejects negative/miscounted indices (`Vulkan_api_kernel.ml:90-114`).
