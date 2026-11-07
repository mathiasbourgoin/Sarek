---
name: roster-spec
type: spec
status: live
feature: Device-only vectors, ranged and stream-ordered copies, pinned host memory, stream launches and events
brief: briefs/device-buffers-async-copy-intake.md
date: 2026-10-10
version: 1.0.0
---

# Spec — device-buffers-async-copy

Driven by a downstream LLM inference engine (stream MoE expert weights host→GPU into a
device-resident cache while the GPU computes). Every design decision below that
was not dictated by the brief was **decided autonomously by the agent** under
the owner's delegation of 2026-10-10; each is marked *(agent decision)*.

## Clarifications

| Q | A |
|---|---|
| Where does the new API live? | `Spoc_core.Transfer` (copies, streams, events, pinned memory, device-only creation), `Spoc_core.Vector.is_device_only`, `Execute.run`/`run_vectors ?stream`. Backends gain `Framework_sig.BACKEND.Async` plus stream/event wrap/unwrap. *(agent decision: Transfer already owns streams and buffer allocation; Vector cannot depend on Transfer.)* |
| How does a stream created in one place reach a buffer closure built elsewhere? | Through extensible `Framework_sig.stream_ref`/`event_ref` with per-backend wrap/unwrap — the existing `kargs = ..` pattern. A backend's `Stream.t` is a fresh abstract type at each first-class-module unpacking, so no other route type-checks. |
| Device->device ranged copy? | Dropped *(agent decision)*: optional in the brief, and a typed core API would need a third cross-unpacking handle (buffer refs). `Framework_sig.ASYNC` does not carry it either. |
| Offsets: elements or bytes? | Typed copies take element offsets; `copy_bytes_to_device` / `copy_ptr_to_device` take byte offsets (an expert block mixes int32 nibbles and f32 scales). Backend `Async` functions take bytes. |
| What is a device-only vector on Native/Interpreter? | Device memory *is* host memory there, and Native kernels read `vec.host` through `Vector.get`. The vector keeps ordinary host storage and is device-resident (`GPU dev`); ranged copies are memcpy, streams are synchronous. Host element access therefore works on CPU devices and refuses on real devices — documented. *(agent decision)* |
| Who keeps async copy sources alive? | Core does: every async copy roots its Bigarray source/destination and the destination vector in a per-stream ledger, released by `synchronize_stream`, or by `synchronize_event`/a completed `event_query` for the work recorded before that event. Raw-address variants are the caller's obligation. |
| Default-stream semantics of `?stream` omitted? | Synchronous: the call returns when the copy is complete (same as today's transfers). |
| Which backends implement? | HIP and CUDA (CUDA/C and CUDA/PTX share `Cuda_plugin_base`): everything. Native, Interpreter: everything with synchronous/memcpy semantics; pinned = plain host memory. OpenCL, WebGPU (and Vulkan and Metal before `specs/vulkan-async-api.md` and `specs/metal-ranged-copies.md`): streams/events keep existing behaviour; every new `Async` operation raises `Backend_error (Plugin Feature_not_supported)` naming the backend and the operation. |
| Float16 pinned buffers? | Refused: ctypes cannot build a Float16 Bigarray over a foreign pointer. Use a `char`/`int16` buffer and the byte-copy variant. *(agent decision)* |

## User Stories

### US-1: Device-only vectors (Priority: P0)
As the downstream engine's tier manager, I want to allocate a 20 GB expert cache that lives
only in VRAM so that host RAM is not spent mirroring it.
**Why this priority**: without it the downstream engine cannot fit the cache in a 125 GB host alongside the RAM tier.
**Scope**: does NOT cover SoA (transparent) vectors or multi-device migration of device-only vectors.
**Independent Test**: create a device-only vector on HIP, check no host storage exists, use it as a kernel argument, read a range back explicitly.
**Acceptance Scenarios**:
1. **Given** a HIP device, **When** `Transfer.create_device_only dev Vector.float32 (1 lsl 20)` is called, **Then** `Vector.is_device_only v = true`, `Vector.location v = GPU dev`, and no host Bigarray is allocated.
2. **Given** a device-only HIP vector filled by `copy_to_device`, **When** it is passed to `Execute.run_vectors` as an input, **Then** the kernel reads the copied values (no host→device auto-transfer happens).
3. **Given** a device-only HIP vector, **When** `Vector.get v 0`, `Transfer.to_cpu v`, `Vector.set`, `Vector.fill`, `Vector.copy` or `Vector.sub_vector` is called, **Then** each raises an error naming the vector as device-only and pointing at `Transfer.copy_to_host`; nothing is allocated.
4. **Given** a device-only Native vector, **When** a Native kernel writes it and the caller reads it with `Transfer.copy_to_host`, **Then** the values match the kernel's output.

### US-2: Ranged host↔device copies (Priority: P0)
As the I/O engine, I want to copy n elements from a host buffer into a device vector at an element offset, synchronously or on a stream, so that experts land in cache slots.
**Why this priority**: the expert cache is slot-addressed; whole-buffer copies from offset 0 cannot fill slot k.
**Scope**: does NOT cover strided/2-D copies.
**Independent Test**: copy distinct ramps into offsets 0, k, n-m of a device vector, read the whole vector back, compare element-wise.
**Acceptance Scenarios**:
1. **Given** a device vector of 1000 int32 zeros, **When** `copy_to_device (sub src 0 10) ~dst ~dst_offset:990` is called, **Then** `copy_to_host ~src:dst ~src_offset:0` of 1000 elements shows the 10 values at 990..999 and zeros elsewhere.
2. **Given** the same vector, **When** `dst_offset = 995` with 10 elements, or `dst_offset = -1`, **Then** `Invalid_argument` is raised before any device call and the device contents are unchanged.
3. **Given** a char Bigarray holding 4 int32 values' bytes, **When** `copy_bytes_to_device src ~dst ~dst_byte_offset:8`, **Then** elements 2..5 of the int32 device vector hold those values.
4. **Given** a vector whose host copy is authoritative (`CPU` or `Stale_GPU`), **When** a ranged copy targets it, **Then** it is refused (call `to_device` first) — a partial device write would leave neither copy authoritative.
5. **Given** an ordinary vector in `Both dev`, **When** a ranged host→device copy completes, **Then** its location is `Stale_CPU dev` and a later `Vector.get` pulls the device copy back (auto-transfer intact).

### US-3: Pinned host memory (Priority: P1)
As the I/O engine, I want page-locked host buffers exposed as Bigarrays so that `pread`/O_DIRECT reads land in memory the GPU can DMA from asynchronously.
**Why this priority**: async copies from pageable memory are staged through a driver bounce buffer; pinning is the documented path to overlap.
**Scope**: does NOT cover NUMA placement or write-combined memory.
**Independent Test**: allocate a pinned buffer, fill it, async-copy to device, read back, compare; `is_pinned` true; free it.
**Acceptance Scenarios**:
1. **Given** a HIP device, **When** `alloc_pinned dev Bigarray.char (256 lsl 20)` is called, **Then** a 256 MiB Bigarray is returned, `is_pinned ba = true`, and it can be written and copied.
2. **Given** a pinned buffer with an async copy still pending on a stream, **When** `free_pinned ba` is called, **Then** it is refused; after `synchronize_stream`, `free_pinned` succeeds and `is_pinned ba = false`; a second `free_pinned` is refused.
3. **Given** an ordinary Bigarray, **When** `register_host dev ba` then `unregister_host ba`, **Then** both succeed on HIP; `free_pinned` on a registered (not allocated) buffer is refused.
4. **Given** an OpenCL/Vulkan/Metal device, **When** `alloc_pinned` is called, **Then** `Backend_error (Plugin (Feature_not_supported _))` is raised naming the backend.

### US-4: Stream launches and event ordering (Priority: P0)
As the runtime layer loop, I want to launch kernels on a compute stream that waits on an event recorded after an async copy on a copy stream, so that copy and compute overlap and are correctly ordered.
**Why this priority**: this is the overlap the downstream expert streaming exists for.
**Scope**: does NOT cover priorities, graph capture, or host callbacks.
**Independent Test**: copy stream A copies x into a device-only vector, records event e; compute stream B waits on e and runs `y = x + 1`; result matches.
**Acceptance Scenarios**:
1. **Given** streams A and B on HIP and event e, **When** `copy_to_device ~stream:a src ~dst:x ~dst_offset:0; record_event e a; stream_wait_event b e; run_vectors ~stream:b ...; synchronize_stream b`, **Then** y = src + 1 element-wise.
2. **Given** an event recorded on a stream with queued work, **When** `event_query e` is called before and after `synchronize_event e`, **Then** after the sync it returns `true`.
3. **Given** a stream created on device D1, **When** it is passed to a copy or launch targeting device D2, **Then** the call is refused naming both devices.
4. **Given** Native, **When** the same program runs, **Then** it produces the same result (streams synchronous, events complete immediately).

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-1 | Native kernels read `vec.host` through `Vector.get`; a storage-less device-only vector cannot run there. | On CPU devices the vector keeps host storage (it *is* device memory). Host access works there; documented as a CPU-device property, not portable behaviour. |
| C-2 | US-1 | A GC finalizer frees device buffers of unreachable vectors; a dropped device-only vector with a copy in flight would be freed under the DMA. | The async ledger roots the destination vector until the stream/event completes. |
| C-3 | US-1 | `free_buffer`/`free_all_buffers` drain to host before freeing; a device-only vector has no host. | They discard device data for device-only vectors (documented: freeing a device-only vector destroys its contents) and leave it unusable: every later host or device operation refuses. |
| C-4 | US-2 | Partial device writes break the whole-vector location model. | Ranged writes require the device copy to be current (`GPU`/`Both`/`Stale_CPU` on that device) and leave `Stale_CPU dev` (or `GPU dev`). Otherwise refused. |
| C-5 | US-2 | The async source can be collected or reused while DMA reads it. | Core ledger roots Bigarray sources; raw-address variant documents the caller contract. Reuse (overwrite) while in flight is the caller's ordering problem — documented. |
| C-6 | US-3 | A Bigarray over foreign memory has no finalizer; after `free_pinned` it dangles. | Explicit lifetime, like `munmap`: documented; `free_pinned` refuses while copies are pending and refuses double free. No finalizer (sub-arrays could outlive the parent). |
| C-7 | US-4 | HIP null-stream semantics: blocking streams synchronise with the legacy default stream, so any sync transfer on the default stream serialises A and B. | Documented; the downstream pattern (device-only vectors, no auto-transfers) issues no default-stream work. Not changed here. |
| C-8 | US-4 | Auto-transfers in `run_vectors ~stream` stay synchronous on the default stream. | Kept (existing semantics intact); vectors already resident cost nothing. Documented. |
| C-9 | US-2/US-4 | Unbounded growth of the keep-alive ledger if the caller never synchronises. | Released per stream on stream sync and per event (sequence numbers) on event sync/completed query. Documented. |
| C-10 | all | Every backend must now implement the new `BACKEND` members. | `Framework_sig.Async_refused` functor and `Framework_sig.Make_refs` keep refusing backends to a few lines; the refusal is a `Feature_not_supported` error, never a silent no-op. |
| X-1 | US-1 | A new `host_storage` constructor fans out to every match on `vec.host` (the `soa` field note in `Spoc_core_base.ml` chose a field for that reason) and breaks user code matching on the public field. | Constructor kept *(agent decision)*: the point of FR-002 is that EVERY host access refuses, and only an exhaustive match makes the compiler find every site (`unsafe_get`, `iter`, `to_bigarray`, `to_ctypes_ptr`, the jsoo `Vector_jsx`, the Native/Interpreter readers). A field would let a forgotten site read a dummy array. The break for code matching on `vec.host` is declared in CHANGES.md. |
| X-2 | US-1 | FR-002 enumerated only six accessors. | Made exhaustive by X-1: every function that matches `vec.host` gets a refusing `Device_only_storage` arm, `unsafe_*` included (the match already exists; no extra cost). |
| X-3 | US-1 | Native `Memory.alloc` is a separate Bigarray while `is_zero_copy` is always true, so a non-aliasing buffer would make ranged copies no-ops for kernels. | On CPU devices `create_device_only` allocates host storage and takes the buffer through `ensure_buffer`, whose zero-copy path aliases it. Required by FR-017. |
| X-4 | US-1 | On CPU devices `Vector.set` (Interpreter results, host writes) turns `GPU d` into `Stale_GPU d`, which would make ranged writes refuse. | Device-only vectors are device-current by definition: ranged-copy preconditions are not applied to them, and copies leave their location alone. |
| X-5 | US-1 | `Int32.of_int` wraps silently for lengths ≥ 2^31 in launch-length injection. | `create_device_only` refuses lengths outside `1 .. Int32.max_int`; the length injection in `Execute.expand_to_run_source_args` now refuses instead of wrapping (FR-018). A 20 GB cache is several vectors. |
| X-6 | US-4 | Vectors bound to `run_vectors ~stream` are not rooted; the GC finalizer can free them under the kernel. | The stream launch adds its argument vectors to the stream's keep-alive ledger (FR-009 widened). |
| X-7 | US-1 | `free_buffer`/`free_all_buffers` could free under in-flight DMA, and leave `CPU` on a host-less vector. | Freeing a vector the ledger still holds is refused. A freed device-only vector keeps `Device_only_storage`; every later operation refuses with "device-only vector has no device buffer". |
| X-8 | US-2 | No read-side precondition. | `copy_to_host` source must be device-current (`GPU`/`Both`/`Stale_CPU` on that device, or device-only). |
| X-9 | US-2, US-4 | Auto-sync after an async ranged write depends on streams being *blocking* (synchronising with the legacy default stream). | Stated as a requirement (FR-019): `Transfer.create_stream` creates blocking streams; switching to non-blocking streams must first make auto-sync stream-aware. |
| X-10 | US-4 | The downstream pattern never syncs the copy stream; ledger grows; flush/device sync do not release. | Release is transitive: `stream_wait_event b e` records the dependency, and completing b up to that point releases e's stream up to e. `Transfer.flush dev` releases every stream of `dev`. HIP's own kernarg list is unchanged (released by stream/device sync, as today) — documented. |
| X-11 | US-2, US-4 | Ledger keyed by handle address collides after destroy/re-create. | Each core stream wrapper carries a unique id. `destroy_stream` synchronises and releases before destroying. A copy on `default_stream d` is ledgered like any stream. |
| X-12 | US-3 | Registered Bigarrays could be collected while registered; pinned registry keyed by address only. | `register_host` roots the Bigarray until `unregister_host`. The registry key is (address, byte length), so a sub-array, even at offset 0, is refused by `free_pinned`. Use of a buffer after `free_pinned` is undefined, as after `munmap` — documented. |
| X-13 | US-3 | Non-portable pinned memory silently degrades copies from another device/context. | Allocation and registration use the PORTABLE flags (`hipHostMallocPortable`, `CU_MEMHOSTALLOC_PORTABLE`, `*_REGISTER_PORTABLE`). |
| X-14 | US-4 | `Event.create : unit -> t` has no device; HIP `Stream.synchronize` lacks `set_current`. | Core `create_event dev` selects `dev` before creating; core events remember their device; recording or waiting across devices is refused. HIP `Stream.synchronize` now selects the stream's device first (as CUDA's does since backlog-225). |
| X-15 | US-4 | `run_vectors ~stream` and `create_device_only` on refusing backends unspecified. | Both refuse with `Feature_not_supported` before any transfer or allocation (a launch ordered on a stream must not run on the default queue). |
| X-16 | all | Native/Interpreter `support` must not advertise overlap they do not have; error type for mismatches. | Native/Interpreter report `ranged_copy`, `pinned_host`, `host_register`, `stream_events` true and `async_copy`, `stream_launch` false; their `~stream` operations are accepted and complete synchronously, which satisfies any ordering. Caller errors (bounds, device/stream mismatch, wrong location) raise `Invalid_argument`; missing capabilities raise `Feature_not_supported`. The refuse-vs-emulate call (capability-model §6) was delegated by the owner: decided autonomously by the agent. |
| X-17 | FR-016 | The BACKEND signature change breaks out-of-tree plugins. | Declared in CHANGES.md with the migration (`Async_refused`, `Make_refs`, `run_source_on_stream` via `refuse_stream_launch`: about 20 lines). |
| X-18 | all | Thread-safety across domains. | Ledger and pinned/registered tables are mutex-protected and never touched from a GC finalizer. Vector location/buffer mutation keeps its existing single-domain-per-vector assumption — documented. |

## Functional Requirements

#### Device-only vectors
- **FR-001** [US-1]: `Transfer.create_device_only` MUST allocate the device buffer and MUST NOT allocate a host Bigarray or custom host storage on a non-CPU device.
- **FR-002** [US-1]: Every host-side read, write or whole-vector read-back of a device-only vector on a non-CPU device MUST raise an error naming the vector as device-only; it MUST NOT allocate host memory.
- **FR-003** [US-1]: A device-only vector MUST be accepted as a kernel argument by `Execute.run_vectors` on its own device without any host→device transfer.

#### Ranged copies
- **FR-004** [US-2]: The ranged copy functions MUST copy exactly the requested range at the requested offset and MUST leave every other element unchanged.
- **FR-005** [US-2]: Out-of-range offsets/lengths MUST raise `Invalid_argument` before any device call.
- **FR-006** [US-2]: A ranged device write MUST be refused when the vector's device copy is not current on the target device.
- **FR-007** [US-2]: After a ranged device write to a vector with a host copy, a subsequent auto-synced host read MUST observe the device contents.

#### Async and keep-alive
- **FR-008** [US-2, US-4]: With `~stream`, a copy MUST be enqueued on that stream and return without waiting; without it, MUST complete before returning.
- **FR-009** [US-2, C-5]: SPOC MUST keep Bigarray sources/destinations and the destination vector reachable until the stream or a later-recorded event is synchronised.

#### Pinned memory
- **FR-010** [US-3]: `alloc_pinned` MUST return page-locked memory on HIP and CUDA and ordinary host memory on Native/Interpreter.
- **FR-011** [US-3]: `free_pinned` MUST refuse buffers not returned by `alloc_pinned`, already freed, or referenced by pending async work.

#### Streams and events
- **FR-012** [US-4]: `Execute.run_vectors ~stream` MUST enqueue the launch on that stream (HIP, CUDA); on Native/Interpreter it MUST run synchronously.
- **FR-013** [US-4]: `stream_wait_event b e` MUST order all later work on b after the work recorded by e.
- **FR-014** [US-4]: Stream/device mismatches MUST be refused.

#### Capability refusals
- **FR-015** [all]: On OpenCL, Vulkan, Metal and WebGPU every new operation MUST raise `Feature_not_supported`; it MUST NOT silently fall back to a synchronous whole-buffer copy.
- **FR-016** [all]: Existing transfer, auto-sync and launch behaviour for callers that do not use the new API MUST be unchanged.

#### Additions from challenge round 1
- **FR-017** [US-1, X-3]: On CPU devices the device-only vector's device buffer MUST alias its host storage.
- **FR-018** [US-1, X-5]: Vector lengths above `Int32.max_int` MUST be refused at device-only creation and at launch-length injection, never wrapped.
- **FR-019** [US-2, X-9]: Streams created by `Transfer.create_stream` are blocking by default. `~non_blocking:true` creates a stream with no implicit synchronisation with the default stream; the implicit read-back and default-stream launch then settle it on the host (`Transfer.settle_vector`). Amended by the non-blocking stream change.
- **FR-020** [US-4, X-10]: Completing a stream MUST release, transitively, the keep-alive entries of the work it waited on through events.
- **FR-021** [US-1, X-7]: Freeing device memory still referenced by pending async work MUST be refused.

## Acceptance Criteria

- AC-1 [US-1]: HIP device-only vector: no host storage, kernel argument works, host access refused → `test_device_buffers` HIP cases pass.
- AC-2 [US-1, C-1]: Native device-only vector: kernel writes, `copy_to_host` reads → Native cases pass.
- AC-3 [US-2]: ranged copies at offsets 0, middle, end; out-of-range refused; bytes variant → pass on Native and HIP.
- AC-4 [US-2, C-4]: ranged write onto `CPU`-located vector refused; onto `Both` leaves `Stale_CPU` and auto-sync reads device data.
- AC-5 [US-3]: pinned alloc/copy/free lifecycle including refusals → pass on HIP; Native allocation works.
- AC-6 [US-4]: copy stream + event + compute stream gives correct result on HIP and Native.
- AC-7 [US-3, C-6]: bandwidth report (pinned vs pageable, 256 MiB) prints GB/s — information only, not a gate.
- AC-8 [FR-015]: a refusing backend's `Async` raises `Feature_not_supported` (unit test on the refusal functor; no OpenCL device needed).
- AC-9 [FR-016]: full `dune runtest` passes.

## Edge Cases

- EC-1 [US-2]: zero-length source → no-op, no device call, offset still bounds-checked (`0 <= off <= length`).
- EC-2 [US-2]: copy exactly to the last element (`off = length - n`) → accepted.
- EC-3 [US-1]: device-only vector of length 0, or of a custom (record) kind → refused (`Invalid_argument`); typed copies have no Bigarray for record kinds. *(agent decision)*
- EC-4 [US-3]: `free_pinned` on a sub-array of a pinned buffer → refused (not the allocation start).
- EC-5 [US-4]: `event_query` on a never-recorded event → `true` (HIP/CUDA semantics), ledger untouched.
- EC-7 [US-4]: an event recorded twice → the latest record is what waits, queries and releases refer to.
- EC-6 [US-1]: device-only vector migrated to a second device by `run_vectors` → refused (would need a host round-trip).

## Runnable Checks

- CHECK-1 [AC-1, AC-3, AC-4, AC-5, AC-6]: `dune build @sarek-hip/test/e2e-hip` (runs `sarek-hip/test/test_hip_device_buffers.exe`; self-skips without HIP) → exit 0, no `FAIL` line.
- CHECK-2 [AC-2, AC-3, AC-6, AC-8]: `sarek/tests/e2e/test_device_buffers.exe` (in `dune runtest`) → exit 0; full suite on Native/Interpreter (and CUDA when present), refusal checks on OpenCL/Vulkan/Metal devices when present.
- CHECK-3 [AC-7]: `dune build @sarek-hip/test/e2e-hip-bench` → prints pinned and pageable GB/s (information only).
- CHECK-4 [AC-8]: `spoc/framework/test/test_framework_sig.exe` → `Async_refused: OK`, `Make_refs: OK`.
- CHECK-5 [AC-9]: `dune runtest` → exit 0.

(Commands run through `opam exec` in the local opam switch.)

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"device-buffers-async-copy","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-001","lifecycle":"draft","external_sources":[],"depends_on":[]}
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
{"record":"requirement","id":"FR-015","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-016","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-017","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-018","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-019","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-020","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-021","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-001","FR-002","FR-003","FR-018","FR-021"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-003","FR-017"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-004","FR-005"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-006","FR-007"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-010","FR-011"]}
{"record":"acceptance-criterion","id":"AC-6","for":["FR-008","FR-009","FR-012","FR-013","FR-014","FR-019","FR-020"]}
{"record":"acceptance-criterion","id":"AC-7","for":["FR-010"]}
{"record":"acceptance-criterion","id":"AC-8","for":["FR-015"]}
{"record":"acceptance-criterion","id":"AC-9","for":["FR-016"]}
{"record":"check","id":"CHECK-1","for":["AC-1","AC-3","AC-4","AC-5","AC-6"]}
{"record":"check","id":"CHECK-2","for":["AC-2","AC-3","AC-6","AC-8"]}
{"record":"check","id":"CHECK-3","for":["AC-7"]}
{"record":"check","id":"CHECK-4","for":["AC-8"]}
{"record":"check","id":"CHECK-5","for":["AC-9"]}
```

## Entities

- `device-only vector`: a `Vector.t` whose data lives only in one device's memory; on non-CPU devices it has no host storage (`host = Device_only_storage`).
- `stream_ref` / `event_ref`: backend-tagged, extensible handles letting core pass a backend stream/event between independently unpacked backend modules.
- `keep-alive ledger`: per-stream list of values SPOC roots while async work referencing them may still run.
- `pinned buffer`: a Bigarray over page-locked host memory returned by `Transfer.alloc_pinned`, freed only by `Transfer.free_pinned`.
