# Tools Support

## Component Inventory

- `tools/device_info.ml`: command-line device diagnostic utility installed as `sarek-device-info`.
- `tools/backend_init.ml`: shared backend initialization helper.
- `tools/backend_{cuda,opencl,vulkan,metal}.{available,unavailable}.ml`: Dune-selected backend init stubs.
- `tools/dune`: Dune library/executable definitions and optional backend selects.

## Per-File Purpose

- `tools/device_info.ml`: initializes all available backends, calls `Device.init`, prints device memory, compute capability, threading, and feature information, then exits with status 0 or 1.
- `tools/backend_init.ml`: initializes native and interpreter plugins unconditionally, then calls optional GPU backend stubs.
- `tools/backend_cuda.available.ml`, `tools/backend_opencl.available.ml`, `tools/backend_vulkan.available.ml`, `tools/backend_metal.available.ml`: each calls the matching `Sarek_*.*_plugin.init`.
- `tools/backend_*.unavailable.ml`: each defines `let init () = ()`.
- `tools/dune`: builds `backend_init` and the public `sarek-device-info` executable, using `select` to avoid hard build-time GPU dependencies.

## Features and APIs

- Formats memory in MB/GB (`tools/device_info.ml:7-12`).
- Prints device ID, name, backend, memory, compute capability, max threads/block, compute units, warp size, and feature flags (`tools/device_info.ml:14-39`).
- Reports no-device guidance for CUDA, OpenCL, Vulkan, Metal, Native, and Interpreter (`tools/device_info.ml:54-63`).
- Exposes Dune public executable name `sarek-device-info` under package `sarek-tools` (`tools/dune`), not `sarek`: `backend_init` links the GPU backend packages, which depend on `sarek`, so placing the tool in `sarek` made a `sarek` <-> `sarek-cuda` package cycle. `sarek-tools` lists the backend packages as opam `depopts`.

## Invariants

- Native and Interpreter should be initialized even without GPU packages (`tools/backend_init.ml:3-10`).
- Optional GPU backend absence should be a no-op rather than a build or runtime error.
- `Device.count ()` and `Device.get i` should agree after `Device.init ()` (`tools/device_info.ml:50-78`).

## Potential Invariant Violations or Bugs

- The "No devices found" path says Native/Interpreter are always available but still exits 1 if `Device.count () = 0` (`tools/device_info.ml:54-63`). If native/interpreter initialization fails, this is useful; otherwise it indicates a deeper invariant break.
- Usage text says `Device.best () selects best device (CUDA > OpenCL > Native)` (`tools/device_info.ml:83-85`), but this ordering is not validated in this tool and may drift from runtime implementation.
- Feature label `CPU` comes from `cap.is_cpu` (`tools/device_info.ml:47`); it says nothing about host-memory sharing, which on OpenCL is the explicit `Host_ptr_buffer` opt-in.

## Performance and Maintainability Risks

- Diagnostic output is human-readable only; there is no JSON mode for CI, support automation, or issue templates.
- Backend initialization has no per-backend error isolation. A failing optional backend init can abort device listing for all other backends unless lower layers catch errors.
- **Updated 2026-07-02 — no longer just duplication, now functional divergence:** `tools/backend_*.available.ml` and `benchmarks/backend_*.available.ml` used to be identical copies; they have since drifted. `benchmarks/backend_cuda.available.ml` (and the OpenCL/Vulkan/Metal siblings) gates plugin `init ()` on `SPOC_DISABLE_GPU`/`SPOC_DISABLE_<BACKEND>` env vars and carries an SPDX header; `tools/backend_cuda.available.ml` has neither — it unconditionally calls `Sarek_cuda.Cuda_plugin.init ()` with no env-var check and no SPDX header. This means `sarek-device-info` cannot be told to skip a backend the way benchmark runs can, and the two "identical" stub sets are no longer identical.

## Related Tests and Checks

- `README.md` instructs users to run `dune exec -- sarek-device-info` for installation verification (`README.md:155-168`).
- The executable is built through normal Dune/package builds via `tools/dune`.
- No dedicated tests were found in `tools/**`.

## Missing Tests

- Build matrix tests for each optional backend present/absent select path.
- Unit or snapshot test for memory formatting and feature rendering.
- Integration test that native/interpreter-only environments still report at least one usable CPU device.
- Error-isolation test for one failing optional backend.

## Concrete Improvement Candidates

- Add `--json` output for support reports and CI diagnostics.
- Wrap each optional backend init with a warning instead of aborting all device discovery.
- Share optional backend stub modules between `tools/` and `benchmarks/` to reduce drift.
- Add an explicit runtime check that Native/Interpreter were registered before `Device.init`.
