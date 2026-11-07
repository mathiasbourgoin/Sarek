---
name: roster-spec
type: spec
status: live
feature: Cheap small host->device writes (staged), documented transfer pitfalls
brief: downstream user findings of 2026-10-10 (tartarus, HIP API log, a register benchmark)
date: 2026-10-10
version: 1.0.0
---

# Spec — transfer-pitfalls

Driven by a downstream LLM inference engine (Qwen3.5-122B decode on an RX 7900 XTX). Decisions not
dictated by the brief were **made autonomously by the agent** under the
owner's delegation of 2026-10-10 and are marked *(agent decision)*. Owner
rule applied throughout: semantics-preserving optimisations may be default;
anything with risk or a behaviour change is an explicit, opt-in API.

## Findings this spec answers (measured, tartarus, 2026-10-10)

| ID | Finding | Status here |
|---|---|---|
| F-1 | `Vector.set` on a vector a kernel has used costs a device→host download (after a device sync) and, at the next launch, a synchronous upload from pageable memory that waits for the GPU queue. Downstream engine: ~27 ms of a ~63 ms GPU budget per token (48 layers × 2 vectors × 8 elements). | Reproduced in SPOC: `bench_hip_small_writes` (rewrite, 300 µs host work, 4 launches): 322 µs/iteration without rewrite, 700 µs with `Vector.set` (spin 1000); 1018 vs 1327 µs (spin 4000). The host work cannot overlap the GPU any more. After this change (three runs, spin 1000): none 322.0–322.5, `Vector.set` 705.2–715.1 (writes 341.3–346.4), `stage_to_device` 330.1–338.9 (writes 9.1–11.5) µs/iteration; raw output `docs/measurements/transfer-pitfalls-2026-10-10/bench_hip_small_writes.txt`. |
| F-1b | The caller-side workaround (reuse one pinned buffer, `copy_to_device ~stream:default`) races: rewriting the buffer while the previous copy is still queued changes what that copy delivers. | Reproduced: the first version of the bench had a `pinned` mode; its output differed from the `Vector.set` mode ("RESULT MISMATCH"). The downstream engine is safe today only because routing reads the GPU first, which drains the queue. |
| F-2a | Registering anonymous memory that was never touched gives slow copies; touching first fixes it. | Slow first passes reproduced, the dependence on touching NOT: `bench_hip_register_host` (8 GiB, 5.5 MiB sync copies) and its first versions, 50 first passes over five preparations (written before registering, untouched, written after registering, populated with `MADV_POPULATE_WRITE`, populated then written): 7 ran at 0.95–6.17 GB/s, spread over every preparation (touched 2, untouched 3, written after 1, populated then written 1); about 26 GB/s otherwise; one repeat pass at 1.66. Downstream `regbench 8 5.6 400 late`: 10.82 GB/s in one run. |
| F-2b | After CPU writes into one 8 GiB registration copies fell to 2.3 GB/s; 64 MiB registrations kept ~26.3. A copy may not span two registrations. | Spanning: reproduced (`hipErrorInvalidValue`, synchronous and stream copies). Slow-after-write: **not reproduced** today: `regbench 8 5.6 400` with an O_DIRECT fill of the registered range 26.30 GB/s (one region) and 25.94 (64 MiB regions); interleaved CPU writes 24.9 GB/s. |
| F-2c | `hipHostRegister`ed pages are not counted as pinned (`VmPin: 0 kB`). | Reproduced: `VmPin: 0 kB` with 8 GiB registered. Also measured: `hipHostRegister` itself faults in every page (VmRSS 0.19 → 8.58 GB for 8 GiB untouched). |

Conditions for all rows: tartarus (Ryzen 9 7950X, 125 GB RAM, RX 7900 XTX,
ROCm 7.2, Linux 7.1.2), 2026-10-10 evening, ~64 GB free RAM, 59 GB in zram
swap, 17 GB tmpfs, other workloads running. Raw output: `docs/measurements/transfer-pitfalls-2026-10-10/`.

## Clarifications

| Q | A |
|---|---|
| Can `Vector.set` itself become cheap by default? | No safe default exists *(agent decision)*. (1) The download is required: `set` is a read-modify-write of a `Stale_CPU` buffer (backlog-181). (2) Avoiding `Stale_CPU` after a launch needs to know the kernel never writes the vector; there is no write analysis, launches without IR exist, and an unsound analysis returns stale data silently. (3) Making the auto-upload asynchronous through staging changes when upload errors surface and adds hidden pinned memory to every upload. So: document the cost, and add an explicit cheap path. |
| Where does the cheap path live? | `Transfer.stage_to_device` beside `copy_to_device`; the ring bookkeeping in a new pure module `Staging_ring` (unit-testable without a device). |
| Why not tell users "alloc_pinned + copy_to_device ~stream"? | F-1b: reusing the pinned buffer races unless the caller tracks completion with events. `stage_to_device` snapshots the source into SPOC-owned staging, so the caller may reuse its source as soon as the call returns. |
| Size limit? | `Transfer.max_staged_bytes` = 64 KiB per call; staging is 1 MiB of pinned memory per device, allocated on first use and kept for the process lifetime *(agent decision)*. Larger transfers belong to `alloc_pinned` + `copy_to_device`. |
| Which stream? | `?stream`, default the device's default stream: ordered after every launch already enqueued there and before every later one (and, HIP/CUDA blocking streams, with streams from `create_stream`). |
| Backends? | HIP, CUDA (`async_copy`): staged and asynchronous. Native, Interpreter, Metal, Vulkan (ranged copies without async copies): a synchronous ranged copy from the source (complete on return, so the snapshot property holds trivially; on Metal it waits for kernels that bind `dst`, on Vulkan for every pending launch of the device). OpenCL, WebGPU: `Feature_not_supported`, like `copy_to_device`. |
| Populate on register_host? | Implemented as opt-in `register_host ~populate:true` (madvise `MADV_POPULATE_WRITE`), then **withdrawn** in QA *(agent decision)*: populated memory had slow first passes too (F-2a), so the option had no demonstrated benefit. Documentation only. |
| A helper registering in regions (F-2b)? | Not added *(agent decision)*: its performance motivation did not reproduce, and the spanning hazard is a one-line rule. Documented on `register_host` with both measurements and their conditions. |

## User Stories

### US-1: Cheap small host→device writes (P0)
As the downstream engine's router, I want to publish 8 expert ids and 8 weights per layer to vectors the next kernels read, without making the host wait for the GPU.
**Independent Test**: loop { stage new values; launch a kernel accumulating them } many times; result equals the host-computed sum; HIP time per iteration close to the no-rewrite case.
**Acceptance Scenarios**:
1. **Given** a HIP device-current int32 vector of 1000, **When** `stage_to_device src ~dst ~dst_offset:k` for k ∈ {0, 500, 990}, **Then** a read-back shows exactly those ranges written and every other element unchanged.
2. **Given** a source Bigarray, **When** it is staged and then overwritten immediately (before any sync), **Then** the device receives the values at call time.
3. **Given** 200 iterations of (stage x_i on the default stream; launch acc += x), **When** synchronised, **Then** acc equals Σ x_i element-wise (ordering and snapshot hold across ring wrap-around).
4. **Given** a source above `max_staged_bytes`, an out-of-range offset, or a host-authoritative (`CPU`) vector, **When** staged, **Then** `Invalid_argument` before any device call.
5. **Given** an OpenCL device, **Then** `Feature_not_supported`.

### US-2: The cost of host writes is visible (P0)
As a SPOC user, I want `Vector.set`/`get` documentation to say what they cost on a vector a kernel has used, and what to use instead.
**Acceptance**: `Vector.get`/`set` docstrings and the core README state the download + synchronous upload, and point at `stage_to_device`.

### US-3: Registering untouched memory (withdrawn)
The `~populate` option this story asked for was implemented and withdrawn (Clarifications, F-2a). Nothing ships for it.

### US-4: Register-host pitfalls documented (P1)
**Acceptance**: `register_host` documents F-2a (intermittent slow first passes, independent of preparation), F-2b (no spanning copies; the slow-after-write observation with its conditions and today's non-reproduction), F-2c (`VmPin: 0`; link to slow copies under pressure unproven).

## Challenges

| ID | Challenge | Resolution |
|---|---|---|
| C-1 | Staging slot reused while its DMA is pending. | Ring entries hold an event recorded after the copy; space is reclaimed only when the event completes; when full the call synchronises the oldest event. |
| C-2 | `head = tail` ambiguity in a ring. | Allocations never make `head` reach the oldest live start from below (strict `<`); an empty ring resets to offset 0. Unit-tested with a randomised model. |
| C-3 | Copy enqueued but event record fails: the slot would be reused under DMA. | On that failure the stream is synchronised before the exception propagates; nothing is committed. |
| C-4 | `dst` freed by the GC finaliser while the staged copy is in flight. | The copy goes through `copy_bytes_to_device ~stream`, which roots `dst` in the ledger; the ring's event queries release it. |
| C-5 | Ledger growth under a long loop without synchronisation. | Bounded by the ring: a range is reclaimed (and its ledger entry released) at the latest when the ring needs its space; stream sync and `flush` release everything. |
| C-10 | Review round 1: a failure between enqueueing the copy and committing its range (event creation, a failing event wait inside reserve) could let a retry reuse the range under the pending DMA. | The event is taken before reserve/copy; `reserve` pops the oldest entry only after its wait succeeded; a record failure synchronises the stream and still re-raises the original exception. |
| C-9 | Found during QA, 2026-10-10: on the Ryzen 7950X iGPU (HIP), 300 writes interleaved with 300 launches on the default stream sometimes left a launch's output at its pre-fill (-5). Not staging: every mode of `stress_hip_staged_launches` failed, and plain HIP C does too. **Cause (2026-10-11):** dirty device-memory writes are lost across host idle gaps on the gfx1036 iGPU; kernels run, but what they wrote into `hipMalloc` memory before a pause reads back stale (data written by `hipMemcpy` was not affected). Upstream drm/amd #5923 (https://gitlab.freedesktop.org/drm/amd/-/work_items/5923; GFX clock gating implicated there, GFXOFF ruled out by our own test; same bug in AMDGPU.jl, workaround `AMD_OPT_FLUSH=0`). Host pauses (GC, sleeps) amplify it; `Gc.full_major` before the loop does not remove it, contrary to the earlier lead. Evidence: `docs/measurements/hip-igpu-lost-launches-2026-10-11`. | Fixed in SPOC: every device buffer on an integrated HIP device is allocated with `hipDeviceMallocUncached` (`Hip_api.Memory`; `SAREK_HIP_IGPU_UNCACHED=0` turns it off, unsafe). Process-wide `AMD_OPT_FLUSH=0` is not used: it would also slow the dGPU. Cost measured at about 8-9% on a copy kernel, none on a read-only sweep (host under load; GEMM-like kernels not measured). `stress_hip_staged_launches` is the regression check, hand-run only with `make stress-427` (one 300-launch run per HIP device, strict on every device); it is not in `dune runtest`, because heavy iGPU load hard-reset the machine on 2026-10-11. Downstream hazard: KV caches and intermediate buffers that a kernel writes and that sit across an idle gap on any integrated GPU, if allocated outside SPOC with plain `hipMalloc`. |
| C-6 | Two domains staging on one device. | Per-device staging mutex around reserve/copy/record/commit. Lock order: staging → ledger/pinned, never the reverse. |

## Functional Requirements

- **FR-001** [US-1]: `stage_to_device` MUST write exactly the requested range and leave every other element unchanged.
- **FR-002** [US-1]: The source MUST be read before the call returns; later writes to it MUST NOT affect the device.
- **FR-003** [US-1]: On HIP and CUDA the call MUST NOT wait for previously enqueued GPU work, except when the 1 MiB staging ring is full (then for the oldest staged copy only).
- **FR-004** [US-1]: The write MUST be ordered on the given stream (default: the device's default stream) after work already enqueued there and before work enqueued after it.
- **FR-005** [US-1]: Sources above `max_staged_bytes`, out-of-range offsets, and vectors whose device copy is not current MUST raise `Invalid_argument` before any device call. Zero-length sources MUST be a no-op after the bounds check.
- **FR-006** [US-1]: A vector in `Both` MUST become `Stale_CPU` (off-CPU), so auto-sync reads the device copy.
- **FR-007** [US-1]: Backends without ranged copies MUST raise `Feature_not_supported`; backends without async copies MUST perform a synchronous ranged copy.
- **FR-008** [US-1]: Staging memory MUST be bounded (1 MiB pinned per device that used it).
- **FR-009, FR-010** [US-3]: withdrawn with `~populate`.
- **FR-011**: `register_host` MUST behave exactly as before (signature unchanged).
- **FR-012** [US-2, US-4]: Docs MUST state the measured costs with their conditions, and label unproven links as such.

## Acceptance Criteria

- AC-1 [FR-001, 002, 004, 005, 006]: suite `test_staged` passes on Native, Interpreter and HIP (and CUDA/Metal when present).
- AC-2 [FR-003, 008]: `Staging_ring` unit tests: no two live ranges overlap under a randomised model; wait only when full; refusal above capacity.
- AC-3 [FR-007]: refusing backends: `check_refused` covers `stage_to_device`.
- AC-4 [FR-011]: existing `register_host` cases of the suite pass unchanged.
- AC-5 [F-1]: `bench_hip_small_writes` prints us/iteration for none / vector_set / staged and exits 1 if the outputs differ — information otherwise.
- AC-6 [F-2a]: `bench_hip_register_host` prints first-pass and repeat GB/s for touched, untouched and written-after-registering memory — information only.
- AC-7: full `dune runtest` passes.

## Runnable Checks

- CHECK-1 [AC-1, AC-4]: `dune build @sarek-hip/test/e2e-hip` → exit 0, no `FAIL`.
- CHECK-2 [AC-1, AC-3, AC-4]: `sarek/tests/e2e/test_device_buffers.exe` (in `dune runtest`).
- CHECK-3 [AC-2]: `sarek/core/test/test_staging_ring.exe` (in `dune runtest`).
- CHECK-4 [AC-5, AC-6]: `dune build @sarek-hip/test/e2e-hip-bench`.
- CHECK-5 [AC-7]: `dune runtest`.

(Through `opam exec` in the local opam switch.)

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"transfer-pitfalls","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-001","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-002","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-003","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-004","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-005","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-006","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-007","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-008","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-011","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-012","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-001","FR-002","FR-004","FR-005","FR-006"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-003","FR-008"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-007"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-011"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-003","FR-012"]}
{"record":"acceptance-criterion","id":"AC-6","for":["FR-012"]}
{"record":"check","id":"CHECK-1","for":["AC-1","AC-4"]}
{"record":"check","id":"CHECK-2","for":["AC-1","AC-3","AC-4"]}
{"record":"check","id":"CHECK-3","for":["AC-2"]}
{"record":"check","id":"CHECK-4","for":["AC-5","AC-6"]}
{"record":"check","id":"CHECK-5","for":["AC-7"]}
```

## Plan

1. `sarek/core/Staging_ring.ml` + `sarek/core/test/test_staging_ring.ml` (TDD: model test first).
2. (`register_host ?populate`: implemented, withdrawn in QA.)
3. `Transfer.stage_to_device`, `max_staged_bytes`; per-device staging.
4. Suite: `test_staged`, `test_staged_ordering`, `test_staged_async`, refusal; run Native/Interpreter, HIP, ZLUDA if available.
5. Benches: `bench_hip_small_writes`, `bench_hip_register_host` (alias `e2e-hip-bench`).
6. Docs: `Vector.get`/`set` docstrings, `register_host`/`alloc_pinned` docstrings, `sarek/core/README.md`, `kb/sarek/runtime/core.md`, `CHANGES.md`.
