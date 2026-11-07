---
name: roster-spec
type: spec
status: live
feature: Cheap repeated kernel launches (launch-preparation memo, cheap compile-cache key, single-buffer HIP kernargs)
brief: briefs/cheap-repeated-launches-intake.md
date: 2026-10-10
version: 1.0.0
---

# Spec — Cheap repeated kernel launches

Driver: a downstream LLM inference engine issues ~495 `Execute.run_vectors` launches per decoded token
on HIP. Per launch, Execute regenerated the kernel source (3–19 µs), re-ran the
capability analysis (0.3–6 µs) and the backend re-digested the source with MD5
for its compile-cache key (1–3 µs), all for an IR that had not changed. HIP
argument marshalling did one ctypes heap allocation per argument. Measurements
are in the intake brief.

## Clarifications

| Q | A |
|---|---|
| What identifies "the same kernel"? | The physical IR record (`==`). `Sarek_ir_types.kernel` is immutable and the PPX hands the same `kirc.body_ir` to every launch. A structurally equal but freshly built IR is a miss and behaves exactly as before. No structural fallback. |
| What does generated source depend on? | IR, backend, `block` (Vulkan reads it), `soa_params` (derived from the arguments), and the global intrinsic/type registries. The memo key is (IR, the backend's `generate_source` closure compared physically, block, soa_params); registries are covered by the documented contract and `clear_launch_cache` (C-4). |
| What does the capability verdict depend on? | IR and the device's `capabilities` record. The memo is keyed on the device id AND re-validates that the cached capabilities record is physically the device's current one (C-3). |
| Which paths are covered? | `Execute.run` (and `run_vectors` through it) on JIT, Direct and Custom backends. `Soa_launch.run_soa` (CUDA/PTX-only SoA path) and `run_source` (caller-supplied source) are out of scope. |
| Is the CUDA argument marshalling changed? | Yes, since package H (C-13): one slot block plus one pointer array per launch, as on HIP. Executed via ZLUDA on an AMD GPU, not on NVIDIA hardware. CUDA also benefits from the Execute memo and the shared key memo. |
| Is a prepared-launch API added? | No (intake decision, by evidence). |

## User Stories

### US-1: Repeated launches skip IR-only work (Priority: P0)
As a host program launching the same Sarek kernel many times, I want launches after the first to skip codegen and capability analysis, so that host enqueue cost approaches the driver floor.
**Why this priority**: it is the dominant measured cost (≈60–70 % of a launch).
**Scope**: does NOT cover IR rebuilt per launch, `run_source`, or `Soa_launch`.
**Independent Test**: launch one kernel twice on a counting probe backend; `generate_source` runs once.
**Acceptance Scenarios**:
1. **Given** a JIT backend and kernel IR `k`, **When** `run_vectors ~ir:k` runs 3 times with the same block, **Then** the backend's `generate_source` is called once and all 3 launches receive a physically identical source string.
2. **Given** the same IR launched with different scalar arguments, **When** launched, **Then** it is a hit and results equal those of an un-memoised launch.
3. **Given** an IR `k'` structurally equal to `k` but physically distinct, **When** launched, **Then** `generate_source` runs for it (miss), with unchanged behaviour.

### US-2: The memo never changes what runs (Priority: P0)
As a SPOC user, I want memoisation to be unobservable apart from speed, so that correctness and diagnostics are unchanged.
**Why this priority**: a stale or shared entry would launch the wrong code silently.
**Scope**: does NOT cover re-registration of intrinsics (US-5).
**Independent Test**: probe backend recording calls; vary each key component.
**Acceptance Scenarios**:
1. **Given** IR `k` launched with block 64 then block 128, **When** launched, **Then** `generate_source` runs once per distinct block.
2. **Given** an int64 kernel and a device whose capabilities lack Int64, **When** launched twice, **Then** both launches raise the capability refusal.
3. **Given** a `generate_source` that raises, **When** launched twice, **Then** both launches raise (nothing memoised).
4. **Given** a wrong-arity argument list, **When** launched after a successful launch of the same IR, **Then** `check_launch_args` still raises.
5. **Given** two `Device.t` with the same id but different `capabilities` records, **When** the same IR is launched on both, **Then** the capability check runs for each.

### US-3: Cheap compile-cache key for a re-presented source (Priority: P1)
As a backend, I want `Compile_cache.make_key` to skip re-digesting a source string it has already digested, so that the per-launch key costs a lookup, not MD5 over the source.
**Scope**: does NOT change the key format.
**Independent Test**: compare the key for a string with the key for a fresh copy of it.
**Acceptance Scenarios**:
1. **Given** a source `s`, **When** `make_key` is called twice with `s` and once with `String.sub s 0 (String.length s)` (a fresh copy), **Then** the three keys are byte-identical.
2. **Given** the empty source, **When** keyed, **Then** the key equals the pre-change key.

### US-4: HIP and CUDA kernel arguments marshalled into one buffer (Priority: P1)
As the HIP and CUDA backends, I want all argument values written into one contiguous host buffer, so that a launch does two allocations instead of one per argument.
**Scope**: HIP (`Hip_api.Kernel.launch`) and CUDA (`Cuda_api.Kernel.launch`, shared by CUDA/PTX and CUDA/C). OpenCL, Vulkan, Metal, Native and Interpreter have no per-argument host cells of this kind and are unchanged.
**Independent Test**: existing HIP e2e tests with mixed argument types still pass; new benchmark result check; CUDA: `sarek-cuda/test/test_cuda_kernargs.ml` (allocation count, slot layout, mixed-type e2e, stream launch with a GC before synchronisation).
**Acceptance Scenarios**:
1. **Given** a kernel with buffer, int32, float32 arguments, **When** launched on HIP, **Then** results equal the reference (bench + e2e-hip).
2. **Given** a kernel with int64/float64 arguments, **When** launched on HIP, **Then** values arrive bit-exact (existing f64/i64 e2e tests).
3. **Given** a stream launch, **When** the GC runs before the stream is synchronised, **Then** the argument buffer is still alive (retained exactly as before until `retire_stream`/`retire_device`).

### US-5: Explicit invalidation (Priority: P2)
As a caller that replaces an intrinsic implementation after kernels have run, I want `Execute.clear_launch_cache ()` to force regeneration.
**Scope**: does NOT detect registry changes automatically.
**Independent Test**: probe backend; launch, clear, launch → 2 codegen calls.
**Acceptance Scenarios**:
1. **Given** a memoised IR, **When** `clear_launch_cache ()` then a launch, **Then** `generate_source` runs again.
2. **Given** `Cache_hooks.notify_clear_all ()` or a device-destroy notification, **When** the next launch runs, **Then** it regenerates (memo cleared by the hook).

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-1 | No stable identity if callers rebuild IR or pass their own `Lazy.t` | Key on the forced IR value's physical identity; rebuilt IR is a miss with unchanged behaviour (US-1 sc.3). Entries are weak in the IR, so misses do not leak. |
| C-2 | US-1 | `Soa_launch.run_soa` calls `generate_source` itself | Out of scope (CUDA/PTX SoA path), unchanged. |
| C-3 | US-2 | Device ids are reused (`Device.reset`, hand-built devices) and verdicts depend on `capabilities` | Capability entry stores the `capabilities` record and is a hit only if it is physically the launching device's record; memo is also cleared on `on_clear_all` and on any `on_device_destroy`. |
| C-4 | US-2 | Codegen reads mutable registries not in the key | Registries are populated at module initialisation and only grow; a missing intrinsic raises and is not memoised. Replacing an implementation after first use is the one case not tracked: documented on `run`/`run_vectors`, remedied by `clear_launch_cache` (US-5). |
| C-5 | US-2 | Ordering of argument-dependent checks | `check_launch_args` runs on every launch first, then `soa_param_names`, then the memo lookup. |
| C-6 | US-2 | Codegen side effects (Vulkan debug log) vanish on a hit; codegen refusals | Debug log only on the miss — accepted, documented. Refusals depend only on key components; failures are never memoised, so they re-raise. |
| C-7 | US-2 | Ephemeron mechanics: hash, equality, which part is weak | Weak key = IR only (review round 1: the backend part of the source key is the physical `generate_source` closure, since a backend can be re-registered under the same name), equality `==`, hash = `Hashtbl.hash` of (kern_name, kern_params) (bounded, plain data). Value = small list of (backend, block, soa_params) → source and (device id, capabilities) → verified; values never reference the IR. The value is one typed slot of the IR's record in `sarek/execute/Ir_memo.ml`, shared with the launch window's `ir_info` and the matrix plan, so a launch hashes its IR once (not at all when its domain's previous launch had the same IR); `clear_launch_cache` makes entries stale by generation instead of resetting a table. |
| C-8 | US-2 | Concurrency (domains) | One mutex around lookup/insert only; codegen runs outside it. A racing duplicate build is allowed; the first installed entry wins and is returned, so callers still share one physical string. A generation counter bumped by clear drops builds that started before a clear. |
| C-9 | all | jsoo build compiles Execute.ml | Only `Mutex`, `Ephemeron`, `Hashtbl`, `Atomic`, `Domain.DLS` and `Type.Id` from Stdlib are used; QA builds the jsoo smoke target. |
| C-10 | US-3 | Physical-string memo hits only for reused strings; `Bytes.unsafe_to_string` aliasing | Only reused strings benefit (Execute now reuses them); others pay MD5 exactly as before. Mutating a string obtained by `Bytes.unsafe_to_string` violates the Stdlib contract; documented as a precondition in `Compile_cache.mli`. |
| C-11 | US-3 | `Compile_cache` becomes stateful | Own mutex, lookup/insert only; weak in the string; key equality covered by a test against a fresh copy. |
| C-12 | US-4 | Slot layout, alignment, retention, zero args | 8-byte aligned slots, each value written with its own C type at its slot start (no endianness assumption); buffer + pointer array retained via `retain_kernargs` exactly as today; zero arguments allocate a 1-slot buffer. |
| C-13 | US-4 | CUDA parity | `Cuda_api.Kernel.launch` uses the same layout as C-12 (`marshal_args`: one `int64` slot block, one pointer array, retained via `retain_kernargs`). Executed via ZLUDA on gfx1100 (not NVIDIA hardware): `test_cuda_kernargs` passes, 3 and 10 argument launches are cheaper (H5b log). Not run on NVIDIA hardware. |
| C-14 | US-5 | Scope of clear; in-flight builds | Clears the Execute memo (the digest memo is content-transparent and needs no clearing); generation counter per C-8; hooked to `on_clear_all` and `on_device_destroy`. |

## Functional Requirements

#### Launch-preparation memo
- **FR-001** [US-1]: `Execute.run` MUST call a JIT backend's `generate_source` at most once per (physical IR, physical `generate_source` closure, block, soa_params) while the entry lives, and MUST pass the same physical source string to every launch that hits.
- **FR-002** [US-1]: `Execute.run` MUST run `check_device_capabilities` at most once per (physical IR, device id, physical capabilities record) while the entry lives.
- **FR-003** [US-2]: `Execute.run` MUST run `check_launch_args` on every launch, before any memo lookup.
- **FR-004** [US-2]: The memo MUST NOT store any failure: a launch whose capability check or codegen raises MUST raise again on the next identical launch.
- **FR-005** [US-2]: The memo MUST NOT keep an IR alive (weak in the IR).
- **FR-006** [US-2]: The memo MUST be safe under concurrent launches from several domains.
#### Compile-cache key
- **FR-007** [US-3]: `Compile_cache.make_key` MUST return byte-identical keys before and after this change for every input, and MUST NOT recompute any digest for a (device, name, options) it has already keyed with a physically identical source string while that string lives.
#### HIP marshalling
- **FR-008** [US-4]: `Hip_api.Kernel.launch` MUST allocate at most two host blocks per launch (value slots, pointer array) and MUST keep them reachable until the stream or device is retired.
- **FR-009** [US-4]: Every argument value MUST reach the kernel bit-identical to the pre-change marshalling.
- **FR-009a** [US-4, C-13]: `Cuda_api.Kernel.launch` MUST meet FR-008 and FR-009 as well (`Kernel.kernarg_blocks_allocated` counts the blocks).
#### Device selection under finalisers (found during implementation)
- **FR-013** [US-4]: Freeing a HIP buffer MUST leave the previously selected device selected, because the free runs from GC finalisers in the middle of launches on other devices. (Found when the cheaper launches moved GC timing: the 2-device benchmark failed with hipErrorInvalidDevice.)
#### Invalidation
- **FR-010** [US-5]: `Execute.clear_launch_cache ()` MUST make the next launch of every IR regenerate source and re-check capabilities; `Cache_hooks.notify_clear_all` and device-destroy notifications MUST have the same effect.
- **FR-011** [US-5]: A build that started before a clear MUST NOT be installed in the memo after it.
#### Measurement
- **FR-012** [US-1]: The repository MUST provide `bench_launch_overhead` reporting host µs per launch (3 and 10 arguments, with and without flush) on HIP and Native.

## Acceptance Criteria

- AC-1 [US-1 happy path]: same IR launched 3× on a counting probe JIT backend → 1 `generate_source` call, physically equal sources.
- AC-2 [US-1, C-1]: physically distinct, structurally equal IR → its own `generate_source` call.
- AC-3 [US-2, C-7]: different block → separate codegen. The soa_params component is non-empty only on CUDA/PTX with SoA-bound vectors, which a probe backend cannot reach; it is verified by review (key includes it), not by a test.
- AC-4 [US-2, C-3]: capability refusal raised on every launch; same id with another capabilities record → re-checked.
- AC-5 [US-2, C-5]: arity mismatch after a hit → `Type_mismatch` raised.
- AC-6 [US-2]: raising `generate_source` → raises on each launch.
- AC-7 [US-3]: keys for `s`, `s` again and a fresh copy of `s` are identical and equal the documented format.
- AC-8 [US-4]: HIP e2e suites and the benchmark's result checks pass.
- AC-8a [US-4, C-13]: `test_cuda_kernargs`: 0, 1, 3 and 10 arguments allocate exactly 2 blocks; slots are contiguous 8-byte cells holding bit-exact values; a mixed int32/float32/int64/float64 launch is bit-exact, and a stream launch keeps its argument blocks retained across a GC until the stream is synchronised (`pending_kernarg_count` is 1 before, 0 after; removing the `retain_kernargs` call turns it red).
- AC-9 [US-5, C-14]: `clear_launch_cache` and `notify_clear_all` → next launch regenerates.
- AC-11 [US-4, FR-013]: free a buffer of device 0 with device 1 selected → device 1 still selected.
- AC-12 [US-2, FR-006, FR-011, review]: a clear inside codegen is not installed; 4 domains launching one IR all receive one physical source.
- AC-13 [US-2, review]: a backend re-registered under the same name generates its own source (the source key is the backend's `generate_source` closure, not its name).
- AC-10 [US-1]: benchmark shows lower host µs/launch on HIP than the baseline in the intake brief.

## Edge Cases

- EC-1 [US-1]: same IR, different scalar args → hit.
- EC-2 [US-2]: same IR on two devices of one backend → one codegen (source is device-independent), two capability checks.
- EC-3 [US-2]: IR garbage-collected → entry reclaimed at some later GC; no timing guarantee.
- EC-4 [US-2]: two domains miss at once → both may generate; the first installed source is returned to both.
- EC-5 [US-3]: empty source → key unchanged.
- EC-6 [US-4]: zero-argument kernel → launches with a 1-slot buffer.
- EC-7 [US-5]: intrinsic re-registered without a clear → previously generated source is reused (documented contract).
- EC-8 [all]: jsoo smoke build → compiles.

## Runnable Checks

- CHECK-1 [AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-9, AC-12, AC-13]: `dune build @sarek/execute/test/runtest` (runs `sarek/execute/test/test_launch_memo.ml`) → exit 0.
- CHECK-2 [AC-7]: `dune build @spoc/framework/test/runtest` (runs the extended `test_compile_cache.ml`) → exit 0.
- CHECK-3 [AC-8]: `dune build @sarek-hip/test/e2e-hip` on a HIP host → all suites pass.
- CHECK-4 [AC-10]: `dune exec sarek-hip/test/bench_launch_overhead.exe` → numbers recorded before/after in the PR.
- CHECK-6 [AC-11]: `dune exec sarek-hip/test/test_hip_free_keeps_current.exe` on a 2-HIP-device host → PASSED (red without the fix: verified).
- CHECK-5 [all]: `dune build` (including the jsoo smoke target) and `dune runtest` → exit 0.

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"cheap-repeated-launches","spec_lifecycle":"draft"}
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
{"record":"acceptance-criterion","id":"AC-1","for":["FR-001"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-001","FR-005"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-001"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-002","FR-004"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-003"]}
{"record":"acceptance-criterion","id":"AC-6","for":["FR-004"]}
{"record":"acceptance-criterion","id":"AC-7","for":["FR-007"]}
{"record":"acceptance-criterion","id":"AC-8","for":["FR-008","FR-009"]}
{"record":"acceptance-criterion","id":"AC-9","for":["FR-010","FR-011"]}
{"record":"acceptance-criterion","id":"AC-10","for":["FR-012","FR-006"]}
{"record":"acceptance-criterion","id":"AC-11","for":["FR-013"]}
{"record":"acceptance-criterion","id":"AC-12","for":["FR-006","FR-011"]}
{"record":"acceptance-criterion","id":"AC-13","for":["FR-001"]}
{"record":"check","id":"CHECK-1","for":["AC-1","AC-2","AC-3","AC-4","AC-5","AC-6","AC-9","AC-12","AC-13"]}
{"record":"check","id":"CHECK-2","for":["AC-7"]}
{"record":"check","id":"CHECK-3","for":["AC-8"]}
{"record":"check","id":"CHECK-4","for":["AC-10"]}
{"record":"check","id":"CHECK-5","for":["AC-1"]}
{"record":"check","id":"CHECK-6","for":["AC-11"]}
```

## Entities

- `launch memo`: Execute's table from physical kernel IR to its generated sources (keyed by the backend's physical `generate_source` closure, block and soa_params) and verified capability records.
- `digest memo`: Compile_cache's table from a physical source string to its MD5 hex digest.
