# OpenCL Backend

## Component Inventory

- `sarek-opencl/README.md`: package documentation and examples.
- `sarek-opencl/dune`: optional `sarek-opencl.plugin` library with `str` dependency; modules are listed at `sarek-opencl/dune:7-28`.
- `sarek-opencl/Opencl_error.ml`: shared backend error helpers.
- `sarek-opencl/Opencl_types.ml`: OpenCL handles, constants, result codes, device info enums, dimensions.
- `sarek-opencl/Opencl_bindings.ml`: dynamic OpenCL loading and raw FFI wrappers.
- `sarek-opencl/Opencl_api.ml`: platforms, devices, contexts, queues, memory, programs, kernels.
- `sarek-opencl/Opencl_plugin_base.ml`: framework backend implementation.
- `sarek-opencl/Opencl_plugin.ml`: backend registration, intrinsic registry, codegen, and external source execution.
- `sarek-opencl/Sarek_ir_opencl.ml`: Sarek IR to OpenCL C code generator.
- `sarek-opencl/test/`: `test_opencl_error.ml`, `test_sarek_ir_opencl.ml`, and test `dune`.

## Per-File Purpose

- `Opencl_error.ml` instantiates `Backend_error.Make`.
- `Opencl_types.ml` models OpenCL scalar constants, memory flags, queue flags, program/build info enums, device types, and result conversion. `string_of_cl_error` maps many unknown values to `CL_ERROR_OTHER` at `sarek-opencl/Opencl_types.ml:501-519`.
- `Opencl_bindings.ml` lazily loads OpenCL from common library names at `sarek-opencl/Opencl_bindings.ml:27-46`, exposes OpenCL 1.x and 2.x queue creation, and returns fallback errors for optional calls such as fill/marker/barrier when symbols are absent.
- `Opencl_api.ml` wraps platform/device enumeration, context/queue creation, buffer allocation and copies, program build log handling, kernel argument binding, and launch.
- `Opencl_plugin_base.ml` caches per-device context/queue state, wraps memory/stream/event/kernel APIs, and binds framework arguments to OpenCL kernel args.
- `Opencl_plugin.ml` selects OpenCL devices, generates source with optional FP64 pragma, compiles kernels, and registers with priority 90.
- `Sarek_ir_opencl.ml` emits OpenCL C with helpers, record/variant definitions, vector length parameters, local memory declarations, and OpenCL intrinsics.

## Features and APIs

- OpenCL platform/device enumeration and selection.
- Context and command queue creation with 2.0 `clCreateCommandQueueWithProperties` and 1.x fallback at `sarek-opencl/Opencl_bindings.ml:139-171`.
- Buffer allocation with optional `CL_MEM_USE_HOST_PTR`, host/device copies, program build with logs, kernel cache, scalar/buffer argument setting, and 3D kernel launch.
- External source language support is OpenCL only at `sarek-opencl/Opencl_plugin.ml:209`.
- Code generation supports scalar/vector parameters, vector length params, arrays, records, variants, shared locals, barriers, atomics, math intrinsics, and FP64 detection.

## Invariants

- Device info queries should check every OpenCL return code before reading output buffers.
- Program objects should be released on every build or kernel-create failure.
- Zero-copy memory must preserve host/device coherency requirements for every device type.
- Kernel argument indices are explicit and should be honored regardless of call order.
- Launch dimensions must match valid OpenCL work dimensions and device limits.

## Potential Invariant Violations and Bugs

- `Device.make_device` ignores the return code from the `CL_DEVICE_MAX_WORK_ITEM_SIZES` query at `sarek-opencl/Opencl_api.ml:202-216`; failed queries can leave default or undefined sizes.
- Allocation computes `size * elem_size` without negative-size or overflow validation at `sarek-opencl/Opencl_api.ml:366-407`.
- Copy APIs use byte counts from one side only and do not validate the other side capacity at `sarek-opencl/Opencl_api.ml:413-491`.
- Zero-copy `device_to_host` is a no-op at `sarek-opencl/Opencl_api.ml:433-451`. This may be valid for some `CL_MEM_USE_HOST_PTR` devices, but host coherency can require finish/map/unmap behavior on others. Marked uncertain.
- If `Program.build` or `Kernel.create` fails in `Kernel.compile`, the OpenCL program created at `sarek-opencl/Opencl_plugin_base.ml:349-356` is not released.
- `device_to_device` is not implemented in the plugin at `sarek-opencl/Opencl_plugin_base.ml:283-289`, even though `clEnqueueCopyBuffer` is bound in `sarek-opencl/Opencl_bindings.ml:243-250`.
- Event timing is fake wall-clock timing. `Event.record` writes only `end_time` and `elapsed` subtracts `start.start_time`, which starts at `0.0`, at `sarek-opencl/Opencl_plugin_base.ml:314-325`.
- `Sarek_ir_opencl.atomic_add` supports two or three args but reports expected count `3` even for the two-arg form at `sarek-opencl/Sarek_ir_opencl.ml:310-328`.
- `SWarpBarrier` emits `sub_group_barrier(CLK_LOCAL_MEM_FENCE)` at `sarek-opencl/Sarek_ir_opencl.ml:524-528` without gating on OpenCL version or subgroup extension support. Marked likely portability bug.
- `DShared` without a static size emits an unsized `__local` declaration at `sarek-opencl/Sarek_ir_opencl.ml:694-700`; this may be invalid inside a kernel body unless it maps to a dynamic local argument. Marked uncertain.

## Performance and Maintainability Risks

- Platform and device enumeration runs each time through `Device.get` rather than using a stable cache at `sarek-opencl/Opencl_api.ml:252-279`.
- Per-device plugin state is cached without release/destruction at `sarek-opencl/Opencl_plugin_base.ml:160-174`.
- `generate_source` catches all exceptions and returns `None` at `sarek-opencl/Opencl_plugin.ml:183-196`, hiding codegen failure reasons.
- README API and codegen descriptions are stale. For example, vector type mapping is documented as `float4`, but implementation maps `TVec` to pointer types at `sarek-opencl/Sarek_ir_opencl.ml:56-68`; README examples reference `Memory.write`, `Queue`, and other names not present in the current API.

## Related Tests

- `sarek-opencl/test/dune:1-5` defines the two Alcotest executables.
- `sarek-opencl/test/test_opencl_error.ml:137-156` covers shared error constructors and formatting.
- `sarek-opencl/test/test_sarek_ir_opencl.ml:192-216` covers literals, operations, simple statements, barriers, declarations, and basic codegen snippets.

## Missing Tests

- Device info failure handling for max work item sizes.
- Build failure cleanup and kernel-create failure cleanup.
- Real event profiling or documented placeholder behavior.
- `device_to_device` behavior, either implemented or explicitly tested as unsupported.
- Out-of-order argument setting with sparse indices.
- Generated OpenCL compile tests for records, variants, FP64, subgroup barriers, shared memory, vector length parameters, and atomics.
- Zero-copy coherency tests on devices where `CL_MEM_USE_HOST_PTR` is supported.

## Concrete Improvement Candidates

- Check and propagate `clGetDeviceInfo` errors for max work item sizes.
- Use `Fun.protect` or explicit cleanup around program build and kernel creation.
- Implement `device_to_device` via `clEnqueueCopyBuffer` with size validation, or document why it is intentionally unavailable.
- Replace wall-clock `Event` placeholders with OpenCL profiling events when queue profiling is enabled.
- Preserve codegen exceptions in logs or return structured error details instead of `None`.
- Gate subgroup barriers and FP64 extensions based on detected support, with tests that assert emitted pragmas.
