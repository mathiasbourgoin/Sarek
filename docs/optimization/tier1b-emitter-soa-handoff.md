# Tier 1b implementation notes — device-side SoA emitter

_Implemented 2026-07-24 on branch `feat/tier1b-emitter`. Records what shipped,
the correctness/benchmark evidence, and the precise Tier 1c handoff._

Companion to [tier1a-soa-pinned-handoff.md](tier1a-soa-pinned-handoff.md)
(host + transfer half) and [opt-spoc-runtime.md](opt-spoc-runtime.md) §1
(SoA) / [opt-ptx-passes.md](opt-ptx-passes.md) (#3 `ld.global.nc`, warp prims).

## What shipped — the PTX emitter's SoA lowering

The emitter-side work Tier 1a deferred: a custom (record) vector kernel
parameter can be lowered as **Structure-of-Arrays**, so a single value written
as `pts.(i).x` compiles to coalesced per-leaf loads instead of a strided packed
access.

- **Opt-in surface (v1): a codegen-level channel.**
  `Sarek_ir_ptx.generate ?soa_params:(string list)` (and
  `generate_with_types`). Naming a vector parameter in `~soa_params` lowers it
  SoA; the default `[]` is all-AoS. This is the deliberate v1 boundary: the
  emitter and its ABI are complete and proven; the *user-facing*
  `Vector.create ~layout:SoA` + automatic launch expansion is Tier 1c (below).
- **Kernel signature** (`Sarek_ir_ptx_kernel.emit_params`): a SoA custom-vector
  param expands to one `.param .u64 param_<name>_soa_<field>` per scalar leaf
  (record declaration order) + one shared `.param .u32 param_sarek_<name>_length`,
  instead of the single AoS `(ptr, len)` pair. Leaf base registers are recorded
  in `reg_alloc.arr_soa`.
- **Element addressing** (`Sarek_ir_ptx_mem`, `emit_soa_*`): each field access is
  a plain coalesced scalar-array access at the leaf's own base
  (`shl`/`mul.wide`-free `cvt`+`shl`+`add`+`ld/st.global.<ty>`), reusing the
  existing scalar path — strictly less code than the AoS aggregate path
  (`emit_agg_elem_addr` + byte-offset folding). Whole-element read/write builds
  the identical `ARecord` SROA binding the AoS path produces, so every
  downstream consumer is unchanged.
- **Dispatch**: the four aggregate element sites (`emit_agg_array_read`,
  `emit_record_field`, `emit_agg_elem_assign`, `emit_elem_field_assign`) branch
  on `is_soa`; the AoS path is untouched.
- **Validation**: `~soa_params` naming a non-record, or a record with a
  nested-record/variant/array/unit field, raises `Ptx_codegen_error` with a
  precise message. v1 = flat records only (matches host `Soa`). Note SoA imposes
  **no inter-field alignment constraint** (each leaf is independently
  contiguous), so it accepts mixed-width records — e.g. `{i32; f64}` — that the
  packed AoS path rejects as misaligned.

### Zero AoS change (by construction)

Every existing caller (all backends via `generate_with_types`) passes no
`~soa_params`, so the AoS path emits byte-identical PTX. `test_ptx_snapshot`
(now 58 cases incl. the ptxas gate) stays green; `dune runtest sarek/tests
spoc/ir` green under ZLUDA. No `Sarek_ir_layout` change (leaves reuse the
existing enumeration — the STOP condition in the task did not fire).

## Correctness evidence

- **Unit / PTX-text** (`sarek/tests/unit/test_ptx_snapshot.ml`, 6 new cases):
  SoA field read emits N per-leaf pointers + coalesced loads and no AoS element
  stride (with the AoS compilation of the same kernel proven unchanged);
  whole-element copy; single-field write; mixed-width `{i32; f64}` (s32 + f64
  leaf loads, misaligned-AoS accepted); rejection of non-record and
  nested-record SoA. The `ptxas` gate assembles a SoA kernel.
- **E2e device** (`sarek/tests/e2e/test_soa_emitter_equiv.ml`): the same
  custom-vector `[%kernel]` IR compiled AoS (`run_vectors`) and SoA
  (`generate ~soa_params` + `run_source ~inject_lengths:false` with
  host-transposed leaf vectors) produce identical results to a pure-OCaml
  reference. **PASS** for `point3d` (3× f32) and `dpair` (2× f64) on RX 7900 XTX
  and CPU-as-CUDA under ZLUDA; `point3d`'s AoS leg also passes on
  OpenCL/Vulkan/Native/Interpreter. i32/i64 leaves are covered at the
  PTX-instruction + ptxas-assembly level (device e2e for them is folded into
  Tier 1c once the launch plumbing lands).

| Leaf type combo | PTX markers | ptxas | device e2e (ZLUDA) |
|---|---|---|---|
| f32 × 3 (`point3d`) | ✓ | ✓ | ✓ |
| f64 × 2 (`dpair`) | ✓ | ✓ | ✓ |
| i32 + f64 (`{i;d}`) | ✓ | ✓ (`soa_mixed`) | (Tier 1c) |
| i64 + i32 (`{p;q}`) | ✓ (`ld.global.s64`+`s32`) | ✓ (`soa_long`) | (Tier 1c) |

All three SoA kernels (`soa_field_sum_f32`, `soa_mixed_i32_f64`,
`soa_long_i64_i32`) go through `test_ptxas_assembles`; the i32+f64 and i64+i32
combos additionally have dedicated marker tests. Device e2e for the two integer
combos folds into Tier 1c with the launch plumbing.

## Benchmark — the emitter's coalescing win (`benchmarks/bench_soa_emitter`)

One `wide` (8× f32, 32B stride) custom-vector kernel reading a single field,
compiled both ways; RX 7900 XTX under ZLUDA, median of 50 (effective
single-field GB/s):

| N | AoS ms (GB/s) | SoA ms (GB/s) | speedup |
|---|---|---|---|
| 262 144 | 0.051 (41.1) | 0.038 (55.3) | 1.35x |
| 1 048 576 | 0.066 (127.0) | 0.049 (170.8) | 1.34x |
| 4 194 304 | 0.347 (96.7) | 0.069 (487.0) | **5.03x** |
| 16 777 216 | 0.806 (166.6) | 0.191 (702.8) | **4.22x** |

SoA restores near-peak single-field bandwidth once bandwidth-bound; AoS stays
throttled by the 8× uncoalesced stride. Small N is launch-overhead-bound.
Matches Tier 1a's manual-SoA numbers — **ZLUDA does not eat the win**. (The
emitter produces the same device access pattern as manual N-vector SoA, now from
a single custom-vector argument.)

## Files

- `sarek/codegen/Sarek_ir_ptx_types.ml{,i}` (soa_leaf, arr_soa, is_soa/soa_leaves)
- `sarek/codegen/Sarek_ir_ptx_kernel.ml{,i}` (emit_params SoA branch, ?soa_params)
- `sarek/codegen/Sarek_ir_ptx_mem.ml{,i}` (emit_soa_* addressing)
- `sarek/codegen/Sarek_ir_ptx_expr.ml`, `Sarek_ir_ptx_stmt.ml` (is_soa dispatch)
- `sarek/tests/unit/test_ptx_snapshot.ml`, `sarek/tests/e2e/test_soa_emitter_equiv.ml`
- `benchmarks/bench_soa_emitter.ml`

## Tier 1c handoff — user-facing SoA + read-only cache + warp

### 1. User-facing SoA storage + automatic launch expansion (the plumbing)

The emitter is ready; what remains is making SoA reachable without the
codegen-level `~soa_params` knob:

- **Host storage variant** (`Spoc_core_base.host_storage`, GADT at
  `sarek/core_base/Spoc_core_base.ml:111-120`): add `Custom_storage_soa` holding
  the AoS host buffer (keep host `get`/`set` and the PPX accessors **unchanged**
  — the cheapest design) + the plan + N per-leaf device sub-buffers. Because
  `host_storage` is a GADT, the compiler forces every match site (~25, listed in
  the research notes: `Spoc_core_base`, `Vector`, `Vector_transfer`, `Transfer`,
  `Vector_jsx`) to handle it.
- **`Vector.create ~layout:AoS|SoA`** (`Spoc_core_base.ml:227`): a `~layout` arg
  selecting the variant; `SoA` builds the plan (via `Soa.plan_of_elttype`) and
  allocates N leaf buffers.
- **Transfer** (`sarek/core/Transfer.ml`): `device_buffers` is one-buffer-per-
  device (`:229`); SoA needs N. Either generalise the table or let the SoA
  variant own N leaf scalar `Vector.t` (reuses the scalar transfer path, exactly
  Tier 1a's "each leaf is an ordinary scalar Vector" intent). `to_device`
  scatters AoS→leaves then transfers each; `to_cpu` gathers back.
- **Launch** (`Execute.run` + `expand_to_run_source_args`,
  `sarek/execute/Execute.ml:242`): for a SoA `Vec`, emit N `RSA_Buffer`s (leaf
  order) + one `RSA_Vector_Length`; and, on CUDA/PTX only, pass the SoA param
  names to `generate_source` (thread a `~soa_params` through `Framework_sig`, or
  build a per-launch IR copy). **Gate: SoA activates on `"CUDA/PTX"` only** —
  every other backend transfers the AoS buffer and generates AoS code, the
  documented fallback that guarantees "never wrong data". `of_ctypes_ptr` /
  `sub_vector` stay AoS-only.

This is pure plumbing (no new coalescing) — the roadmap (§1.2c/e) rates it the
bulk of the SoA cost; it was descoped here so the emitter could ship complete
and proven. When it lands, extend `test_soa_emitter_equiv` to drive SoA through
`Vector.create ~layout:SoA` + `run_vectors`, and add the i32/i64 device rows.

> **PRECONDITION — generated param-name namespace (latent today, MUST fix in
> Tier 1c).** The emitter mangles each SoA leaf param as
> `param_<vec>_soa_<field>` (`Sarek_ir_ptx_kernel.emit_params`). This can
> **collide** with a distinct user vector/scalar parameter whose own generated
> name is `param_<vec>_soa_<field>` — e.g. a user param literally named
> `x_soa_y` alongside a SoA vector `x` with field `y`. The reserved `sarek_`
> prefix does **not** cover this infix mangle. It is unreachable today because
> `~soa_params` is only ever passed by the emitter's own tests (never from user
> code), but the moment Tier 1c lets a user opt a vector into SoA it becomes a
> real (silently-wrong-PTX) hazard. Tier 1c **MUST** close it, by either:
> (a) `sarek_`-prefixing the generated SoA params
> (`param_sarek_soa_<vec>_<field>`) so they live in the already-reserved
> namespace and cannot alias a user name; or (b) validating the kernel's param
> names against the generated SoA pattern and rejecting a collision with a
> precise error. Option (a) is preferred (no user-facing rejection, consistent
> with the existing `sarek_<vec>_length` convention). Add a regression test that
> a kernel mixing a SoA vector `x` (field `y`) with a scalar param `x_soa_y`
> compiles to distinct PTX operands.

### 2. `ld.global.nc` for read-only params (roadmap #3 — High, cost S)

Independent of SoA. A single write-set pass over `kern_body` (+ inlined
`hf_body`s) collecting `DParam` array names ever used as an `EArrayWrite` /
`SArraySet` / atomic target; then `emit_array_read` (and the new
`emit_soa_*`/`emit_field_load`) take `~nc:true` for a param-sourced array absent
from the write set, emitting `ld.global.nc.<ty>`. Provably safe (read-only is
static), 1.1–1.3× on multi-pointer-param bandwidth-bound kernels. Not started.

### 3. Warp primitives (roadmap "half-built")

`SWarpBarrier` exists and emits `bar.warp.sync`, but there is no `warp_barrier`
intrinsic: no PPX syntax constructs the statement, so it is emitter-only today;
warp **shuffle** (`shfl.sync`) is not modelled in the IR or emitted. Adding it
is an IR-surface + typer + emitter change (a new intrinsic family), larger than
either item above and not blocking SoA. Scope as its own task.
