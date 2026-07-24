# Tier 1a implementation notes — SoA + pinned memory

_Implemented 2026-07-23 on branch `feat/soa-pinned-tier1a`. This records what
shipped, the correctness/benchmark evidence, and the precise Tier 1b handoff._

Companion to [opt-spoc-runtime.md](opt-spoc-runtime.md) §1 (SoA) and §2 (pinned).
Scope was runtime/layout/host side only; the PTX emitter
(`sarek/codegen/Sarek_ir_ptx_*`) was **not** touched (Tier 1b territory).

## What shipped

### SoA (structure-of-arrays) — host layout + transpose

- **`Spoc_core.Soa`** (`sarek/core/Soa.ml{,i}`): an opt-in SoA storage plan and
  the host-side AoS↔SoA transpose.
  - `plan` / `plan_of_elttype` derive the leaf layout by **reusing
    `Sarek_ir_layout.record_layout`** — the same leaf enumeration the emitter
    and PPX already use, so SoA offsets/sizes/stride match the AoS byte layout
    by construction. v1 accepts **flat records only**; nested records,
    variants, and array/vector fields are rejected with `Soa.Unsupported`
    (variants have no well-defined per-tag SoA split — see opt-spoc-runtime §1
    verdict).
  - `scatter` / `gather` do a bit-preserving transpose between one packed AoS
    ctypes buffer and N per-leaf contiguous buffers (4/8-byte word copy; any
    scalar leaf types).
  - Deliberately layered **below `Vector`** (operates on raw `unit Ctypes.ptr`),
    so each SoA leaf can just be an ordinary scalar `Vector`. That reuses the
    existing scalar transfer / allocation / (future) pinned path with **zero
    backend or codegen change** — the N-scalar-vector bundle *is* the SoA
    storage for the host+transfer tier.

- The one place the roadmap flagged as genuinely harder — growing kernel arity
  so a single custom-vector argument lowers to N base pointers — is **not** done
  here because it is emitter-side. See the handoff below.

### Pinned (page-locked) host memory — CUDA backend API

- **`Cuda_bindings`**: added `cuMemHostRegister_v2` / `cuMemHostUnregister`
  (the `cuMemAllocHost_v2` / `cuMemFreeHost` bindings already existed, as the
  prior exploration pass found).
- **`Cuda_api.Memory`**: `alloc_host` / `free_host` (driver-allocated
  page-locked buffers) and `register_host` / `unregister_host` (page-lock in
  place). The raw pointer feeds the existing
  `host_ptr_to_device` / `device_to_host_ptr` blocking-memcpy path unchanged.

## Correctness matrix (`sarek/tests/e2e/test_soa_aos_equiv`)

AoS and SoA kernels compute the same per-element value and match a pure-OCaml
reference. Verified PASS on **every** device present:

| Framework | Device | Result |
|---|---|---|
| CUDA/PTX (ZLUDA) | RX 7900 XTX | PASS |
| CUDA/PTX (ZLUDA) | Ryzen 9 7950X | PASS |
| OpenCL | RX 7900 XTX (RADV/radeonsi) | PASS |
| OpenCL | Ryzen 9 7950X | PASS |
| Vulkan | RX 7900 XTX (RADV NAVI31) | PASS |
| Vulkan | Ryzen 9 7950X | PASS |
| Native | CPU (32 cores) | PASS |
| Interpreter | CPU (seq + parallel) | PASS |

Plus `sarek/tests/unit/test_soa` (host, no device): plan offsets/sizes/stride
for `point3d` and a mixed-width `{i32;f64}` record, nested/non-record
rejection, and a bit-identical AoS→SoA→AoS round-trip.

## Benchmark numbers

### SoA vs AoS single-field access (`benchmarks/bench_soa_aos`)

Memory-bound copy reading one field of an 8-field (32-byte-stride) record.
RX 7900 XTX under ZLUDA (median of 50 runs, effective single-field GB/s):

| N | AoS ms (GB/s) | SoA ms (GB/s) | speedup |
|---|---|---|---|
| 262 144 | 0.052 (40.3) | 0.043 (48.9) | 1.21x |
| 1 048 576 | 0.066 (127.5) | 0.052 (161.4) | 1.27x |
| 4 194 304 | 0.344 (97.5) | 0.049 (683.2) | **7.00x** |
| 16 777 216 | 0.789 (170.1) | 0.193 (695.9) | **4.09x** |

SoA restores near-peak bandwidth (~680–700 GB/s effective) once the problem is
large enough to be bandwidth-bound; AoS stays throttled at ~100–170 GB/s by the
8× uncoalesced stride. Small N is launch-overhead-bound (~1.2×). Numbers have
run-to-run variance; the large-N coalescing win is the stable signal.

### Pinned vs pageable transfer (`benchmarks/bench_pinned_transfer`)

**Blocked on the environment.** ZLUDA v7-preview.3 implements neither
`cuMemAllocHost` nor `cuMemHostRegister` — both return `CUDA_ERROR_OTHER`. The
benchmark probes for support and, finding none, reports pageable bandwidth only
(RX 7900 XTX): ~28 GB/s H2D/D2H at ≥16 MB, dropping to ~10 GB/s at 1 MB. The
pinned API is correct CUDA per the driver docs; the ~2× pinned-vs-pageable
comparison simply cannot be measured on this hardware/driver and needs a stock
NVIDIA driver (or a ZLUDA build that implements pinned host memory).

> **Update 2026-07-24:** the emitter half of this handoff (item 1 below) is
> now implemented — see
> [tier1b-emitter-soa-handoff.md](tier1b-emitter-soa-handoff.md). The kernel
> signature grows to N base pointers and element addressing is per-leaf
> coalesced, driven by `Sarek_ir_ptx.generate ~soa_params`. Items 2–3 (the host
> `Custom_storage_soa` variant + `Transfer`/`Kernel_args` N-buffer plumbing)
> remain as Tier 1c in that doc.

## Tier 1b handoff — device-side single-vector SoA (emitter)

The host+transfer tier above lets a user *manually* run SoA today (bundle N
scalar vectors, transpose with `Soa`). To make a **single** custom-vector value
transparently SoA-backed — so kernel source keeps writing `pts.(i).x` and the
compiler lowers it to coalesced per-leaf loads — needs three emitter-side
changes, all in files this task was scoped **out** of:

1. **Kernel signature** (`sarek/codegen/Sarek_ir_ptx_kernel.ml:69-93`): a
   custom-vector parameter currently lowers to `(ptr, len)`. Under SoA it must
   lower to `(ptr_0, …, ptr_{N-1}, len)` — N base pointers sharing one length.
   This is the ABI-arity change flagged in opt-spoc-runtime §1.2(c) and touches
   `Kernel_args` / `RSA_Buffer` / `bind_to_kargs` binding
   (`sarek/core/Transfer.ml`) to bind N buffers per vector argument.
2. **Element addressing** (`sarek/codegen/Sarek_ir_ptx_mem.ml`):
   `emit_agg_elem_addr` + `emit_field_load/store`'s byte-offset folding is
   *deleted* for the SoA case and replaced by, per leaf, the plain scalar-array
   addressing that already exists in the same file (`mul.wide.u32 idx,
   leaf_size; add.u64 r_base_leaf`). A record-of-N-scalars becomes N
   independent coalesced scalar accesses with a shared index — **less** emitter
   code, not more.
3. **Storage variant** (`sarek/core_base/Spoc_core_base`): a `Custom_storage_soa`
   host-storage variant holding the N leaf sub-buffers, so `Vector.create
   ~layout:SoA` can pick it and `Transfer` can alloc/transfer/free N device
   buffers. `Soa.plan` already provides the exact leaf list this needs; the SoA
   scatter/gather here becomes the host get/set path.

Guardrails to preserve (already reflected in `Soa`): flat records only for v1;
AoS stays the compatible default (external `.cu`/`.ptx` via `Execute.run_source`
are hand-written against the packed layout); whole-record-copy kernels can
regress under SoA (N cache lines vs 1) so the layout must stay an explicit,
workload-driven per-vector choice, never a blanket default flip.

## Files

- `sarek/core/Soa.ml`, `sarek/core/Soa.mli`
- `sarek-cuda/Cuda_bindings.ml`, `sarek-cuda/Cuda_api.ml` (`Memory` pinned API)
- `sarek/tests/unit/test_soa.ml`, `sarek/tests/e2e/test_soa_aos_equiv.ml`
- `benchmarks/bench_soa_aos.ml`, `benchmarks/bench_pinned_transfer.ml`
