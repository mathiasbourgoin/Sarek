# GPU Backend Packages

This knowledge base covers the optional Sarek GPU backend packages:

- `sarek-cuda/`
- `sarek-opencl/`
- `sarek-vulkan/`
- `sarek-metal/`

Generated documentation under `gh-pages` and benchmark generated descriptions were intentionally excluded from this pass.

## Component Map

All four packages follow the same broad shape:

- `README.md`: package documentation and examples.
- `dune`: optional library packaging and module list.
- `*_error.ml`: shared `Backend_error.Make` instantiation.
- `*_types.ml`: backend-specific handles, enums, result codes, and ctypes structures.
- `*_bindings.ml`: dynamic library loading and raw FFI declarations.
- `*_api.ml`: higher-level device, memory, stream/event, and kernel APIs.
- `*_plugin_base.ml`: implementation of the SPOC framework backend signature. CUDA/OpenCL/Metal share `module type Framework_sig.PLUGIN_BASE` (`spoc/framework/Framework_sig.ml:362-493`); Vulkan does not.
- `*_plugin.ml`: source generation, external-source execution, intrinsic registry, and backend registration.
- `Sarek_ir_*.ml`: Sarek IR to backend kernel language code generation; shared variant/struct emission (`gen_variant_def`/`gen_variant_def_glsl`/`mangle_name`) is factored into `Sarek_ir_codegen` in the `spoc.ir` library.
- `test/`: Alcotest suites for shared error wrappers and code generation.

## Backend Docs

- [CUDA](cuda.md)
- [OpenCL](opencl.md)
- [Vulkan](vulkan.md)
- [Metal](metal.md)
- [Shared patterns and test gaps](shared-patterns.md)

## High-Risk Cross-Cutting Findings

- Kernel argument index handling is inconsistent. CUDA appends and ignores the supplied index in `sarek-cuda/Cuda_plugin_base.ml:268-281`; Vulkan buffer args are assigned sequential bindings in `sarek-vulkan/Vulkan_api_kernel.ml:351-354`; Metal stores indices but discards them before execution in `sarek-metal/Metal_plugin_base.ml:406-416`.
- Runtime buffer copy paths generally trust caller-provided sizes. CUDA copies by source size in `sarek-cuda/Cuda_api.ml:211-238`; OpenCL copies by host/destination size in `sarek-opencl/Opencl_api.ml:413-491`; Vulkan copies by host byte size or explicit byte size in `sarek-vulkan/Vulkan_api_memory.ml:409-543`; Metal uses host dimensions or silently truncates device-to-device copies in `sarek-metal/Metal_plugin_base.ml:253-288`.
- Device and kernel caches are not thread-safe and usually keep stale resources after destroy. CUDA keeps destroyed contexts in `device_cache` after `Device.destroy` at `sarek-cuda/Cuda_api.ml:164-178`; Vulkan has the same pattern for device/default-stream caches in `sarek-vulkan/Vulkan_api_device.ml:30` (destroy at `sarek-vulkan/Vulkan_api_device.ml:311`) and `sarek-vulkan/Vulkan_api_stream.ml:86-93`; OpenCL and Metal keep per-device state caches without cleanup in `sarek-opencl/Opencl_plugin_base.ml:160-174` and `sarek-metal/Metal_plugin_base.ml:153-172`.
- Event timing is mostly placeholder behavior except CUDA. OpenCL uses wall-clock-only records in `sarek-opencl/Opencl_plugin_base.ml:314-325`; Vulkan reports `0.0` elapsed in the event wrapper; Metal records wall-clock fields in `sarek-metal/Metal_plugin_base.ml:312-325`.
- Tests focus on small code generation fragments and shared error constructors. They do not compile generated kernels or exercise real runtime launch, buffer transfer bounds, cache invalidation, out-of-order argument setting, records/variants, or backend-specific language validation.

## Priority Improvement Areas

1. Add deterministic tests for argument binding order and replacement semantics across all backends.
2. Add size and overflow validation to allocation and copy APIs before invoking FFI.
3. Fix stale cache entries on device/kernel destruction, or document backend objects as process-lifetime resources and remove misleading destroy APIs.
4. Compile representative generated kernels in CI where toolchains are available, even if runtime execution remains opt-in.
5. Update READMEs to match actual APIs and generated code; several README examples reference names or generated snippets that do not exist in the current source.
