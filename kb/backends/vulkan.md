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
- `compile_glsl_to_spirv_cli` accepts `entry_point` but ignores it in the CLI command at `sarek-vulkan/Vulkan_api_base.ml:41-56`.
- `Device.destroy` destroys Vulkan handles but leaves cached device/default stream state in process refs at `sarek-vulkan/Vulkan_api_device.ml:311` (device cache at `sarek-vulkan/Vulkan_api_device.ml:30`) and `sarek-vulkan/Vulkan_api_stream.ml:86-93`.
- Staging buffers map memory at `sarek-vulkan/Vulkan_api_memory.ml:176-177`, but staging cleanup frees memory without `vkUnmapMemory` at `sarek-vulkan/Vulkan_api_memory.ml:422-423`, `sarek-vulkan/Vulkan_api_memory.ml:455-456`, `sarek-vulkan/Vulkan_api_memory.ml:489-490`, and `sarek-vulkan/Vulkan_api_memory.ml:523-524`.
- Allocation computes `size * elem_size` without validation at `sarek-vulkan/Vulkan_api_memory.ml:190-192` and `sarek-vulkan/Vulkan_api_memory.ml:306-313`.
- Copy APIs do not validate requested byte counts against source and destination capacities at `sarek-vulkan/Vulkan_api_memory.ml:409-543`.
- `device_to_device` is not implemented at `sarek-vulkan/Vulkan_api_memory.ml:545-547`, even though Vulkan buffer copy commands are already used for staging transfers.
- Buffer binding count is inferred by regex-counting occurrences of `binding = N` at `sarek-vulkan/Vulkan_api_kernel.ml:127-143`, then descriptor layout bindings are created as dense `0..num_bindings-1` at `sarek-vulkan/Vulkan_api_kernel.ml:144-172`. Sparse binding numbers or comments can produce invalid layouts.
- Pipeline creation hardcodes entry point `"main"` at `sarek-vulkan/Vulkan_api_kernel.ml:233`, ignoring the requested kernel name.
- `compile_cached` omits the kernel name from its in-memory key at `sarek-vulkan/Vulkan_api_kernel.ml:314`.
- `set_arg_buffer` ignores the supplied index and assigns sequential bindings at `sarek-vulkan/Vulkan_api_kernel.ml:351-354`.
- Push constant setters append bytes sequentially and do not bounds-check the 128-byte buffer at `sarek-vulkan/Vulkan_api_kernel.ml:356-388`.
- Generated GLSL declares vector length push constants at `sarek-vulkan/Sarek_ir_glsl.ml:823-849`, but runtime buffer binding only records buffers at `sarek-vulkan/Vulkan_api_kernel.ml:351-354`. Unless the higher framework separately passes vector lengths as scalar args in matching order, generated shaders can see zero lengths. Marked likely bug requiring framework trace.
- `launch` ignores the runtime block size at `sarek-vulkan/Vulkan_api_kernel.ml:394-397`; block size is baked into generated GLSL, but direct external source launch can mismatch user expectations.
- The command buffer is waited and fence-reset at `sarek-vulkan/Vulkan_api_kernel.ml:461-472`, then begun at `sarek-vulkan/Vulkan_api_kernel.ml:476-490` without an explicit command-buffer reset. Marked likely Vulkan state bug.
- Generated GLSL maps `TInt64` and `TFloat64` to `int64_t` and `double` at `sarek-vulkan/Sarek_ir_glsl.ml:147-149`, but the emitted header at `sarek-vulkan/Sarek_ir_glsl.ml:784-795` does not enable int64/fp64 extensions. README examples show extensions that implementation does not emit.
- `SNative` is emitted as a comment instead of executable native GPU code at `sarek-vulkan/Sarek_ir_glsl.ml:682-685`.

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

- Replace regex binding inference with structured binding metadata from codegen or SPIR-V reflection.
- Include `name` in cache keys and use the requested entry point consistently across shaderc, CLI compilation, and pipeline stage creation.
- Reset command buffers before re-recording or allocate one-time command buffers per launch.
- Track mapped staging memory and unmap before freeing.
- Emit required GLSL extensions for int64/fp64 based on IR usage and device support.
- Implement `device_to_device` with `vkCmdCopyBuffer` and size validation.
- Add bounds checks for push constant writes and derive push constant layout from the same metadata as codegen.
