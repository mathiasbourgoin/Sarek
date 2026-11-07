---
name: roster-spec
type: spec
status: live
feature: Device-only vectors, ranged copies, pinned host memory and streams/events on the Metal backend
brief: briefs/metal-ranged-copies-intake.md
date: 2026-10-10
version: 1.0.0
---

# Spec — metal-ranged-copies

Follow-up to `specs/device-buffers-async-copy.md` (#423), which defined the
API and implemented it on HIP, CUDA, Native and Interpreter while Metal
refused. Driven by a downstream LLM inference engine on an Apple M4 (32 GB unified
memory): a per-layer MoE expert cache of ~48 device-only vectors filled by
ranged copies. Decisions not dictated by the brief were **decided autonomously
by the agent** under the owner's delegation of 2026-10-10 and are marked
*(agent decision)*.

## Clarifications

| Q | A |
|---|---|
| Storage mode of a device-only vector? | `MTLResourceStorageModeShared`, like every other Metal buffer in SPOC *(agent decision)*. On Apple silicon CPU and GPU share one physical memory: a shared buffer's `contents` is GPU memory, so a ranged copy is one `memcpy` at full memory bandwidth. Private storage would need a shared staging buffer plus a blit encoder, i.e. the same bytes moved twice, with no bandwidth gain for GPU reads on unified memory. "Device-only" is a SPOC property (no host Bigarray mirror), not a Metal storage mode. |
| Maximum buffer size? | The device's `maxBufferLength` (queried per device). An allocation above it is refused before `newBufferWithLength:` with a memory-allocation error naming the requested size and the limit, instead of the opaque "returned null". Callers split larger caches (the downstream engine does: ~48 buffers). |
| Real async (command buffers that complete after the call returns)? | **Launches: yes** (scope addition requested by the lead on 2026-10-10, after measuring 240 us per launch and 8.6 tok/s for the downstream engine's 0.8B decode on ladon; first version of this spec said no). Dispatches are appended to one open command buffer per queue, committed when the queue's previous command buffer has completed or after `SPOC_METAL_BATCH` (default 32) dispatches *(agent decision: sweep 8–512 on the 0.8B decode, 32 within noise of the best)*. **Copies: no**: a copy is a CPU memcpy into shared storage and completes before returning (`async_copy = false`). |
| Stream launches? | Implemented: `run_source_on_stream` appends to the stream's own `MTLCommandQueue`; `support.stream_launch = true`. |
| Ordering of a copy against a kernel? | A per-queue table of pending command buffers records the MTLBuffers each binds (`Metal_api.Pending`). Every host access to a buffer waits for the pending command buffers that bind it (committing an open one first); a launch waits for pending work on OTHER queues binding its buffers; within one queue Metal's default tracked-hazard mode orders command buffers and serial dispatches. A copy issued before a launch is visible to it (CPU writes to shared storage before `commit`). Freeing needs no wait: a command buffer retains the buffers it binds. |
| Pinned host memory? | Page-aligned host memory from `posix_memalign` with the OS page size (16 KiB on Apple silicon) *(agent decision)*. Unified memory has no DMA engine to page-lock for: a copy from it is the same `memcpy` as from pageable memory. Page alignment keeps a future `newBufferWithBytesNoCopy:` wrap possible. Documented in the backend and README. |
| `register_host`? | Accepted as a no-op on the memory, tracked by core as pinned (same as Native) *(agent decision)*: there is nothing to page-lock for the GPU on unified memory, and refusing would make portable downstream code diverge for no benefit. |
| Events? | `record` captures the newest command buffer of the stream (committing the open one); `synchronize` waits on it through `Pending` (a failure is reported once); `event_query` reads its status and reports a failure; `elapsed` uses max(record time, GPUEndTime); `stream_wait_event` adds nothing (dependencies are through buffers, already ordered). |
| What still refuses on Metal? | Nothing new from `Framework_sig.ASYNC`. Raw-pointer kernel arguments, PTX, and the other existing refusals stay. OpenCL, Vulkan, WebGPU keep `Async_refused`. |

## User Stories

### US-1: Device-only vectors on Metal (Priority: P0)
As the downstream engine's tier manager on an M4, I want `Transfer.create_device_only` to allocate an MTLBuffer with no host mirror so that a 14 GiB expert cache does not cost another 14 GiB of RAM.
**Why this priority**: the downstream expert cache on ladon cannot start without it.
**Scope**: does NOT cover private storage or wrapping existing host memory.
**Independent Test**: create a device-only int32 vector on Metal, fill it by ranged copy, use it as kernel input and output, read back.
**Acceptance Scenarios**:
1. **Given** the Metal device of an M4, **When** `Transfer.create_device_only dev Vector.int32 1000`, **Then** the call succeeds, `Vector.is_device_only v = true`, `Vector.location v = GPU dev`.
2. **Given** device-only x filled with 3*i and device-only y, **When** `y = x + 1` runs via `Execute.run_vectors`, **Then** `copy_to_host` of y gives 3*i+1 for all 1000 elements.
3. **Given** a device-only Metal vector, **When** `Vector.get`, `Vector.set`, `Vector.fill`, `Vector.to_bigarray` or `Transfer.to_cpu` is called, **Then** `Invalid_argument` is raised.
4. **Given** a request whose byte size exceeds the device's `maxBufferLength`, **When** the buffer is allocated, **Then** a `Backend_error` names both sizes and no MTLBuffer is created.

### US-2: Ranged copies at offsets (Priority: P0)
As the I/O engine, I want host→device and device→host copies at element/byte offsets into an existing Metal vector so that expert k lands in slot k.
**Why this priority**: slot-addressed cache.
**Scope**: does NOT cover device→device ranged copies (no public API).
**Independent Test**: the suite's `test_ranged` on a Metal device.
**Acceptance Scenarios**:
1. **Given** 1000 int32 zeros, **When** ramps are copied at offsets 0, 500, 990, raw bytes at byte offset 408 and a raw pointer at byte offset 2800, **Then** read-back shows exactly those ranges and zeros elsewhere.
2. **Given** the same vector, **When** offset 995 with 10 elements or offset -1, **Then** `Invalid_argument`, contents unchanged.
3. **Given** an ordinary vector in `Both dev`, **When** a ranged write lands, **Then** location becomes `Stale_CPU dev` and `Vector.get` returns the written value.
4. **Given** a 256 MiB device-only char vector, **When** a 256 MiB ranged copy runs, **Then** it completes and its bandwidth is reported (information only).

### US-3: Streams, events and stream launches (Priority: P1)
As the downstream engine's runtime, I want the HIP/CUDA stream code path (copy on stream a, event, b waits, kernel on b) to run unchanged on Metal so that one code path serves both machines.
**Why this priority**: avoids a Metal-specific branch in the downstream engine.
**Scope**: does NOT provide overlap inside one domain.
**Independent Test**: the suite's `test_streams_pinned` on Metal.
**Acceptance Scenarios**:
1. **Given** pinned x data and streams a, b, event e, **When** copy on a, record e on a, b waits e, launch on b, synchronize b, **Then** y = x + 1 and `event_query e = true`.
2. **Given** a launch on a stream of another device, **When** run, **Then** `Invalid_argument` (core check, unchanged).
3. **Given** a destroyed stream or event, **When** used, **Then** `Invalid_argument` (core check, unchanged).

### US-6: Asynchronous launches (Priority: P0, scope addition 2026-10-10)
As the downstream engine's decoder (~495 launches per token), I want a Metal launch to cost microseconds of host time, not a GPU round trip, so that decode on ladon is not bound by launch latency.
**Why this priority**: 240 us/launch capped Qwen3.5-0.8B decode at 8.6 tok/s.
**Scope**: does NOT make copies asynchronous.
**Independent Test**: many dependent launches with no synchronisation, then one host read; launch-overhead bench before/after.
**Acceptance Scenarios**:
1. **Given** x = 0 (1 Mi int32), **When** 400 alternating launches y = x + 1, x = y + 1 run with no sync and x is read, **Then** every element is 400.
2. **Given** a launch y = x + 1 just enqueued, **When** x is overwritten by a ranged copy, **Then** y holds the old x + 1 and x the new values.
3. **Given** two streams, **When** 200 launches alternate between them on the same buffers with no sync, **Then** the result is 200.
4. **Given** ordinary vectors, **When** 100 unsynchronised launches run and `Vector.get` reads, **Then** the auto-transfer read-back sees the final value.

### US-4: Pinned and registered host memory (Priority: P1)
As the I/O engine, I want `alloc_pinned`/`register_host` to work on Metal so that the same buffers serve `pread` and copies on both machines.
**Why this priority**: portability of the downstream engine's I/O layer.
**Scope**: no page-locking semantics on unified memory.
**Independent Test**: alloc, check page alignment, copy from it, free; register/unregister a Bigarray.
**Acceptance Scenarios**:
1. **Given** Metal, **When** `alloc_pinned dev Bigarray.int32 4096`, **Then** `is_pinned` is true and the address is a multiple of the OS page size.
2. **Given** a pinned buffer referenced by an in-flight stream copy, **When** `free_pinned` before the stream is synchronised, **Then** refused (core ledger, unchanged); after synchronisation it frees.
3. **Given** an address not returned by `alloc_pinned`, **When** the backend `free_pinned` is called, **Then** `Invalid_argument`, nothing freed.

### US-5: Other backends unchanged (Priority: P0)
As a SPOC user on OpenCL/Vulkan/WebGPU, I want the existing capability refusals to stay, so that nothing silently falls back.
**Why this priority**: capability model.
**Scope**: no behaviour change outside sarek-metal except refusal message text.
**Independent Test**: `test_device_buffers.exe` and `dune runtest` on tartarus.
**Acceptance Scenarios**:
1. **Given** an OpenCL device, **When** `create_device_only`, **Then** `Feature_not_supported`.
2. **Given** tartarus, **When** `dune runtest`, **Then** exit 0.

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-1 | Shared storage on a discrete-GPU Mac (Intel Mac + AMD) lives in system memory and is slow for the GPU. | SPOC's Metal backend already uses shared storage for every buffer; device-only vectors do not change that. Apple silicon is the target; documented. |
| C-2 | US-3 | `stream_launch = true` may mislead callers about copies. | Launches on a stream are asynchronous on the stream's queue (US-6); `async_copy = false` says copies complete on return. |
| C-3 | US-2 | A copy from another domain into a buffer a kernel is reading races. | Each copy waits for pending command buffers binding its buffer (FR-016); the wait is not atomic with the copy, so two domains acting on the same buffers at the same moment must order themselves, as on every backend. |
| C-4 | US-1 | `newBufferWithLength:` may also fail below `maxBufferLength` (memory pressure). | Kept: the existing null check raises `memory_allocation_failed`. |
| C-5 | US-4 | `posix_memalign` memory is not freed by the GC. | Freed only by `free_pinned`; core's registry refuses double free; backend table refuses unknown addresses. |

### Adversarial review (independent sub-agent, 13 challenges, all resolved)

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| A-1 | US-1 | Oversize refusal runs under core's `Gpu_memory.with_retry` (two GCs before re-raise); test must go through `create_device_only`, not the backend. | Accepted cost: the original error is re-raised, once per failed setup. Test calls `Transfer.create_device_only` with `maxBufferLength + 1` chars. |
| A-2 | US-1 | A failed command buffer was silent; `total_global_mem` hard-coded 4 GiB. | FR-012: a command buffer ending in status Error raises `kernel_launch_failed` with the NSError description at the first wait that observes it (launches are asynchronous, US-6), naming every kernel of the batch; failures pending at exit are reported on stderr. FR-013: `total_global_mem` = `recommendedMaxWorkingSetSize`. |
| A-3 | US-1/3 | `device_states` unguarded; `run_source_on_stream` device choice. | `device_states` guarded by a mutex. `run_source_on_stream` compiles for the current device like `run_source` (the plugin cannot see a stream's device: `Stream.t` is abstract there); `Kernel.launch` refuses a stream of another device than the kernel's. Core already refuses a stream of another device before launching. |
| A-4 | US-2 | No 256 MiB test, no aliases, EC-1 untested. | Aliases `e2e-metal` and `e2e-metal-bench` added and declared in `scripts/unwired-targets.tsv` (no macOS runner); zero-byte copy at offset = size tested. |
| A-5 | US-2 | Backend bounds check duplicates core's. | Defensive, as Native's (`Host_async.check_range`); raw-address callers of the backend are not the only route. Kept. |
| A-6 | US-3 | Messages and docs listing async backends become wrong (Framework_sig, Transfer, Execute, kb, suite headers). | FR-014: every such text names Metal. |
| A-7 | US-3 | `stream_events = true` vs "real ordering semantics"; `Event.elapsed` meaningless across streams. | Events capture the newest pending command buffer of their stream (US-6); `event_query` reads its status, reporting a failure through `Pending` (once). `Event.elapsed` subtracts, per event, the later of its record time and its command buffer's `GPUEndTime` (`CLOCK_UPTIME_RAW`); it used host record times. |
| A-8 | US-3 | Cross-device stream check not exercisable on a single-GPU M4. | Core check, exercised on the CPU backends by `test_device_buffers.exe`. Accepted. |
| A-9 | US-3 | Ledger keep-alives still hold vectors until sync. | Intended: core semantics are backend-independent; documented on the synchronous-stream entity. |
| A-10 | US-4 | Pinned table must be global and locked; test address 4096n is unsound. | Global table keyed by address under a mutex; test frees the address of an ordinary Bigarray. |
| A-11 | US-4 | `register_host` with `byte_size <= 0`. | Refused with `Invalid_argument`, as Native. |
| A-12 | US-5 | `test_device_buffers.ml` routes Metal to `check_refused`. | Routed to `Suite.run_on`; run on ladon. |
| A-13 | US-1 | Over-release if `free` runs twice. | Pre-existing precondition: core removes the buffer from the vector before freeing, so the finaliser does not see it again. Unchanged. |

## Functional Requirements

#### Device-only vectors (US-1)
- **FR-001** [US-1]: The Metal backend MUST report `support.ranged_copy = true` so `Transfer.create_device_only` allocates an MTLBuffer with no host Bigarray.
- **FR-002** [US-1]: Metal buffer allocation MUST refuse a byte size greater than the device's `maxBufferLength` with a `Backend_error` naming the requested size and the limit, before calling `newBufferWithLength:`.
- **FR-003** [US-1]: Device-only Metal buffers MUST use `MTLResourceStorageModeShared`.

#### Ranged copies (US-2)
- **FR-004** [US-2]: `Async.host_to_device`/`device_to_host` MUST copy exactly `byte_size` bytes between the host address and `contents + offset`, and MUST raise `Invalid_argument` when `[offset, offset+byte_size)` is outside the buffer.
- **FR-005** [US-2]: A copy MUST have completed when the call returns, with or without a stream.

#### Streams and events (US-3)
- **FR-006** [US-3]: `run_source_on_stream` MUST enqueue the launch on the given stream's command queue, never on the default queue.
- **FR-007** [US-3]: `event_query` MUST be true exactly when the newest command buffer of the stream at record time has finished (or there was none); `synchronize_event` MUST wait for it; `stream_wait_event` MUST NOT raise.
- **FR-008** [US-3]: The Metal backend MUST report `stream_launch = true`, `stream_events = true`, `async_copy = false`.

#### Pinned memory (US-4)
- **FR-009** [US-4]: `alloc_pinned` MUST return OS-page-aligned memory of at least `byte_size` bytes, refusing `byte_size <= 0`; `free_pinned` MUST refuse addresses it did not return.
- **FR-010** [US-4]: `register_host`/`unregister_host` MUST succeed without touching the memory.

#### Asynchronous launches (US-6, scope addition)
- **FR-015** [US-6]: A Metal launch MUST return without waiting for the GPU.
- **FR-016** [US-6]: Every host access to a Metal buffer (whole or ranged copy in either direction, read-back) MUST first wait for the pending command buffers that bind it.
- **FR-017** [US-6]: A launch MUST first wait for pending command buffers on other queues that bind any of its buffers.
- **FR-018** [US-6]: `Device.synchronize` (hence `Transfer.flush`) MUST wait for all pending work; `Stream.synchronize` and `Stream.destroy` for the stream's queue.

#### Robustness (adversarial review)
- **FR-012** [US-1, A-2]: A Metal command buffer that ends with status `Error` MUST raise `kernel_launch_failed`, carrying the error description and the kernels it ran, at the first wait that observes it, exactly once.
- **FR-013** [US-1, A-2]: `capabilities.total_global_mem` MUST report the device's `recommendedMaxWorkingSetSize`.
- **FR-014** [US-5, A-6]: Refusal messages and docs that list the backends implementing ranged copies, pinned memory or stream launches MUST include Metal.

#### Unchanged elsewhere (US-5)
- **FR-011** [US-5]: OpenCL, Vulkan and WebGPU MUST keep refusing every `ASYNC` operation with `Feature_not_supported`.

## Acceptance Criteria

- AC-1 [US-1]: Metal suite `test_device_only` → all ok (device-only y = x + 1).
- AC-2 [US-1, FR-002]: allocation above `maxBufferLength` → `Backend_error` naming both sizes.
- AC-3 [US-2]: Metal suite `test_ranged` → 0 mismatches; boundary refusals.
- AC-4 [US-3]: Metal suite `test_streams_pinned` and `test_destroyed` → all ok.
- AC-5 [US-4]: pinned address page-aligned; unknown-address free refused.
- AC-6 [US-2]: 256 MiB host→device and device→host bandwidth printed (information).
- AC-8 [A-2]: `total_global_mem` printed on ladon equals `recommendedMaxWorkingSetSize` (> 4 GiB on a 32 GB M4).
- AC-9 [US-6]: `test_async_launches` → all ok; removing the buffer wait or the cross-queue wait makes it FAIL (prove-red).
- AC-10 [US-6]: launch overhead and the downstream engine's 0.8B decode tok/s measured before/after on ladon, decode tokens identical.
- AC-7 [US-5]: `dune runtest` on tartarus exit 0; `test_device_buffers.exe` OpenCL/Vulkan refusal checks pass.

## Edge Cases

- EC-1 [US-2]: `byte_size = 0` at offset = buffer size → accepted, no-op.
- EC-2 [US-1]: length × elem_size > `maxBufferLength` but length ≤ Int32.max_int → FR-002 refusal.
- EC-3 [US-4]: `alloc_pinned` with byte_size not a multiple of the page size → rounded allocation is still ≥ byte_size, aligned.

## Runnable Checks

- CHECK-1 [AC-1, AC-3, AC-4, AC-5]: on ladon, `dune build @sarek-metal/test/e2e-metal` → `test_metal_device_buffers.exe` exit 0, no FAIL lines.
- CHECK-2 [AC-2]: on ladon, the same executable's oversize-allocation check → ok.
- CHECK-3 [AC-6]: on ladon, `dune build @sarek-metal/test/e2e-metal-bench` → prints GB/s lines (information, not a gate).
- CHECK-5 [AC-9]: on ladon, `test_metal_device_buffers.exe` async checks ok; mutation runs FAIL.
- CHECK-6 [AC-10]: manual — `bench_metal_launch_overhead.exe` and the downstream engine's `decode --greedy 64` on ladon, numbers in the PR.
- CHECK-4 [AC-7]: on tartarus, `dune runtest` exit 0 and `sarek/tests/e2e/test_device_buffers.exe` exit 0.

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"metal-ranged-copies","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-001","lifecycle":"draft","external_sources":[],"depends_on":["device-buffers-async-copy/FR-001"]}
{"record":"requirement","id":"FR-002","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-003","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-004","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-005","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-006","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-007","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-008","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-009","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-010","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-011","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-012","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-013","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-014","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-8","for":["FR-012","FR-013"]}
{"record":"requirement","id":"FR-015","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-016","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-017","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-018","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-9","for":["FR-015","FR-016","FR-017","FR-018"]}
{"record":"acceptance-criterion","id":"AC-10","for":["FR-015"]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-001","FR-003"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-002"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-004","FR-005"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-006","FR-007","FR-008"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-009","FR-010"]}
{"record":"acceptance-criterion","id":"AC-6","for":["FR-004"]}
{"record":"acceptance-criterion","id":"AC-7","for":["FR-011","FR-014"]}
{"record":"check","id":"CHECK-1","for":["AC-1","AC-3","AC-4","AC-5"]}
{"record":"check","id":"CHECK-2","for":["AC-2"]}
{"record":"check","id":"CHECK-3","for":["AC-6"]}
{"record":"check","id":"CHECK-4","for":["AC-7"]}
{"record":"check","id":"CHECK-5","for":["AC-9"]}
{"record":"check","id":"CHECK-6","for":["AC-10"]}
```

## Entities

- `Pending command buffer`: a committed or open Metal command buffer nobody has waited for, with the MTLBuffers it binds (`Metal_api.Pending`).
- `Metal device-only vector`: a `Vector.t` with `Device_only_storage` whose single buffer is a shared-storage `MTLBuffer`.
- `Synchronous stream`: a stream whose every enqueued operation has completed when the enqueuing call returns (Native, Interpreter). On Metal launches are asynchronous and copies synchronous. Core's keep-alive ledger still applies: operands stay rooted until the stream or an event is synchronised.

## Runtime lock during waits (R-24)

Every blocking wait on a Metal command buffer (`waitUntilCompleted`, from
`Pending.wait_cmd`, `Event.synchronize` and the flush, which reaches the wait
through `wait_all` / `wait_selected`) calls `objc_msgSend` with the OCaml
runtime lock released, as do the two compiles (`newLibraryWithSource`,
`newComputePipelineState`) and the libc `memcpy` of host/device copies
(`Metal_api.memcpy`; its raw-address arguments are not rooted by the call, so
the caller keeps the memory allocated, as on the other backends), so a domain waiting on the
GPU does not stop the other domains' minor collections. Invariants: the
arguments are C pointers only (no OCaml heap pointer), and the command buffer is
retained by the waiter for the duration of the call. Short messages keep the
plain stub: releasing and reacquiring the lock costs more than they take.
Acceptance: `test_metal_wait_binding` (structural) and, on a Metal device,
`test_blocking_ffi_releases_lock` on alias `e2e-metal` (at least 10 minor
collections by an allocating domain during a ~200 ms `Device.synchronize`).

Measured on Apple M4, macOS 15 (2026-10-12, load 2.7): 199 ms wait, 1000 minor GCs on the other domain with the change, 1 without (origin/main 1808a9a5). Raw: `docs/measurements/metal-runtime-lock-2026-10-12/r24-ladon.txt`.
