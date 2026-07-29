# The cost of adding a scalar width — measured, and why bf16 is not landing yet

**Task:** backlog-88 (`bf16-dsl-element-type`) — "bf16 as a DSL element type, and
make width-addition cheap before a third width is added."

**Date:** 2026-07-27.

**Method:** the cost was not estimated. Each type vocabulary had its constructor
actually added, the tree built, and the compiler errors counted, round by round,
until green. The patch used for the measurement is not part of this change; the
numbers below are what it produced.

**Status of the two halves:**

- **The count and its reduction: delivered.** §1–§4.
- **bf16 itself: NOT delivered, deliberately.** §5. The blocker is not the count.
  It is that `docs/design/f16-dsl-element-type.md` §11.1's central premise —
  "bf16 needs no new machinery … as close to *f16 again, with a different `cvt`*
  as a new width can be" — is **refuted by execution**. See §5.1.

---

## 1. The headline number

> **A usable scalar width touches 40 production files. The compiler asks for 21
> of them. The other 20 must be found by reading.**

Not an estimate. Both halves are measured:

| | how it was obtained | count |
|---|---|---|
| production files a width actually needs | `git show --name-only` on `c93dbf96`, the f16 slice-1 commit, filtered to non-test `.ml`/`.mli` | **40** |
| of those, forced by adding the constructors | added `TBFloat16` to `Sarek_ir_types.elttype`, `BFloat16` to `Sarek_ir_analysis.feature`, `TBFloat16` to `Sarek_ir_ppx.elttype` and `BFloat16` to `Sarek_types.registered_type`; built to green | **21** (20 overlap with the 40, plus `Sarek_capability.ml`, which postdates slice 1) |
| **invisible to the compiler** | set difference | **20** |

And the forced half does not arrive at once. It came in **seven build rounds**,
because each layer's errors are hidden behind the previous layer's failure:

| round | vocabulary | sites | files |
|---|---|---|---|
| 1 | `Sarek_ir_types.elttype` | 7 | 3 (`Sarek_ir_layout` ×5, `Sarek_ir_pp`, `Sarek_ir_analysis`) |
| 2 | (cascade) | 16 | 9 (`Sarek_capability`, PTX ×4+3, CUDA, Metal, OpenCL ×2, `Sarek_ir_inline_vec`, `Soa`) |
| 3 | (cascade) | 4 | 3 (OpenCL, WGSL ×2, GLSL) |
| 4 | (cascade) | 2 | 1 (`test_backend_type_width_totality`) |
| A | `Sarek_ir_ppx.elttype` + `Sarek_types.registered_type` | 2 | 2 |
| B | (cascade) | 2 | 2 |
| C | (cascade) | 5 | 2 (`Sarek_ir_conv`, `test_type_width_totality` ×4) |
| | **total** | **38 sites** | **21 production files** |

**The blast radius is not visible up front.** You cannot grep it, and you cannot
see it after one build. This is the mechanical restatement of §9's
"match-arm counting systematically underestimates a new element type" — except
that the underestimate is now bounded: match-arm counting is accurate about the
21, and says nothing at all about the 20.

## 2. The 20 the compiler never asks for

```
sarek/core_base/Spoc_core_base.ml            sarek/core/Ctypes_ops.ml
sarek/core_base/Spoc_core_base.mli           sarek/core/Memory.ml
sarek/core_base/Spoc_core_base_scalar.ml     sarek/core/Vector.ml
sarek/core_base/Spoc_core_base_scalar.mli    sarek/core/Vector_transfer.ml
sarek/execute/Execute.ml                     sarek-hip/Hip_api.ml
sarek/interp/Sarek_float16.ml                sarek/sarek/Sarek_float16.ml
sarek/interp/Sarek_ir_interp_eval.ml         sarek/interp/Sarek_ir_interp.ml
sarek/plugins/interpreter/Interpreter_plugin_base.ml
sarek/plugins/native/Native_plugin_base.ml   sarek/ppx/Sarek_core_primitives.ml
sarek/ppx/Sarek_native_gen_base.ml           sarek/ppx/Sarek_native_intrinsics.ml
sarek/codegen/Sarek_ir_ptx_types.mli
```

This is not a random 20. It is the FFI layer, the host storage layer, the
exec-arg dispatch, the conversion primitives and the interpreter narrowing —
**exactly the list of places the f16 post-mortem says its expensive defects
were** (`f16-dsl-element-type.md` §3.3, §6.4, §9). Restated:

> None of the four round-1 must-fixes and neither of the two soundness bugs
> were in a match arm.

They were all in the 20.

### 2.1 The asymmetry that explains why

`Execute.ml` contains the two halves of one dispatch, forty lines apart:

- `exec_arg_of_vector`'s **`get`** matches on `Vector.kind v` — total. Adding a
  host scalar kind **does** force an arm here (measured: it is in the round-1
  error set of the host-kind experiment).
- `exec_arg_of_vector`'s **`set`** matches on the `(typed_value, Vector.kind)`
  **pair**, and ends in a catch-all. Nothing can force an arm into it.

That is why f16 shipped with a `get` arm and no `set` arm, and why "an f16 kernel
could not run on the Interpreter **device** at all" was discovered by a reviewer
rather than by the build. A pair-match with a catch-all is
**gates-that-cannot-fail pattern #4 (no-op arm)** applied to a dispatch table.

## 3. The reduction

Two sweeps already existed, and they cover the ends of the pipeline while
meeting nowhere:

```
 source type ──▶ IR element type ──▶ device type string
 └──── test_type_width_totality ────┘
                 └──── test_backend_type_width_totality ────┘
```

Neither says anything about the third edge, which is where the bytes live:

```
                 IR element type ◀──▶ HOST scalar_kind ──▶ exec-arg dispatch
```

`sarek/tests/unit/test_host_ir_width_agreement.ml` closes it, with the same
construction the other two use — a wildcard-free total successor chain over the
`scalar_kind` GADT, plus pinned exemption sets. It asserts:

1. **width agreement** — `Vector.elem_size k = Sarek_ir_layout.scalar_size t` for
   every host kind with an IR counterpart. `scalar_size` is the denominator the
   *backend* sweep already checks device types against, so this is what makes the
   three segments one chain instead of three self-consistent fragments;
2. **exec-arg totality** — every host scalar kind either completes a
   store-and-load round trip through `Execute.exec_arg_of_vector`'s `get`/`set`,
   or appears in a pinned refusal list. This is the only instrument that can see
   a missing arm in the catch-all half of §2.1;
3. **pinned exemptions** — the kinds with no IR counterpart (`Char`, `Complex32`)
   and the kinds the dispatch refuses (`Char`, `Complex32`) are each pinned
   exactly.

**What it changes about the count.** It does not shrink the 40. It moves work
from the invisible 20 into the forced 21: a new host `scalar_kind` now fails to
**compile** two total matches in the sweep, and `probe_of` cannot be satisfied by
a stub because each arm must produce a real value *of that kind's own element
type*. The sweep then *executes* the exec-arg path that no constructor addition
can reach. The next width still costs 40 files — but the two defects that cost
f16 the most review time (a missing `set` arm; a host/IR width disagreement) both
become red instead of silent.

`Char` is in the pinned set as evidence rather than as an excuse: `Vector.char`
is one byte, the IR has no one-byte element type, and `char` used to lower to
`Ir.TInt32` and stride the buffer at 4×. That is a width defect this edge would
have caught, recorded in the file it belongs to.

## 4. Reds observed (prove-a-gate-can-fail)

Every claim below was executed on this worktree, not reasoned.

| # | mutation | result |
|---|---|---|
| R1 | add `TBFloat16` to `Sarek_ir_types.elttype` | `test_backend_type_width_totality` **fails to compile**, naming `TBFloat16` as the unmatched case — the backend sweep is genuinely constructor-driven |
| R2 | add `BFloat16` to `Sarek_types.registered_type` | `test_type_width_totality` **fails to compile** (4 sites), same property for the front half |
| R3 | f16 `set` arm in `Execute.ml` made to reject floats (the pre-fix state) | new sweep **FAIL**: *"the set of host scalar kinds the exec-arg dispatch REFUSES changed. expected: Char, Complex32 / actual: Char, Complex32, **Float16**"* — the defect is named |
| R4 | `Sarek_ir_layout.scalar_size TFloat16` 2 → 4 | new sweep **FAIL**: *"**Float16**: the host stores 2 byte(s) per element (Vector.elem_size) but the IR lays the same element out in 4 byte(s)"* |
| R5 | delete the `Complex32` arm of `probe_of` | **fails to compile**, warning 8 as an error — establishing that the sweep's matches are wildcard-free and that a new `scalar_kind` constructor therefore breaks them |
| R6 | add an `Int16` constructor to `scalar_kind` | forced arms in `Spoc_core_base`, `Sarek_ir_interp` ×2, and `Execute`'s `get` — but **not** `Execute`'s `set`. This is the §2.1 asymmetry, measured. |

R3 and R4 are the two that matter: they are the f16 defect and the `char` defect,
reproduced and caught.

## 5. Why bf16 is not in this change

### 5.1 §11.1's premise is refuted by execution

f16's single cheapest property was that **`Bigarray` already had it**. From
`f16-dsl-element-type.md` §3.1, verified there and re-verified here:
`Bigarray.Float16` exists, `Array1.create/get/set` round-trips through binary16
for free, and the host `scalar_kind` arm is
`Float16 : (float, Bigarray.float16_elt) scalar_kind`.

**There is no `Bigarray.Bfloat16`.** Executed on this switch (OCaml 5.3.0): a
probe compiles and runs `Bigarray.Float16` (`3.14159 → 3.140625`), and the
installed `bigarray.mli` enumerates the complete, closed `kind` GADT — 14
constructors, `Float16` among them, no bfloat16 of any spelling.

`scalar_kind` requires a `Bigarray.*_elt` phantom, so bf16 has **no host storage
representation available on f16's route**. Both alternatives are redesigns, not
arms:

- **`Int16_unsigned`** gives `(int, Bigarray.int16_unsigned_elt)`. The host value
  type becomes `int` — `Vector.get` would hand back a raw bit pattern, and every
  bf16↔float conversion moves out of the Bigarray store and into Sarek. The
  storage-genericity of `Vector.get`/`set`/`Soa`/`Execute` is what made f16 cheap,
  and it is exactly what this gives up.
- **`Custom_storage`** does reach the transfer layer, but a `Custom` kind is
  `('a, unit) kind` — **not a `Scalar` kind at all**. It would not flow through
  `scalar_kind`, so it would not flow through the host width table, the exec-arg
  scalar dispatch, or the sweep in §3. `Vector.to_bigarray` raises on it, and
  `(Scalar _, Custom_storage _)` is a refuted case in `Transfer.ml`.

So the correct reading of §11.1 is: bf16 is "f16 again with a different `cvt`"
*from the IR downward*, and something else entirely *from the IR upward*. The
lesson generalises the one §3.3 already recorded — a type existing in the stdlib
says nothing about it existing in every library that mirrors that type's GADT —
one level further: **f16's cheapness came from a stdlib feature, and a sibling
width does not inherit it.**

### 5.2 There is nothing to verify bf16 against

The verification bar for a width on this project is bit-exact agreement between
the interpreter and a real device. For bf16 on this host:

- **CUDA** — `__nv_bfloat16` needs sm_80. There is no NVIDIA device on this box,
  and the only NVIDIA GPU reachable from it is sm_61 (GTX 1070 Max-Q). `nvrtc` would compile it host-side, but a compile is not
  the agreement gate.
- **HIP** — gfx1100 has bf16 in its ISA and `hiprtc` is available, so this is the
  one plausible backend. But CUDA and HIP **share `Sarek_ir_cuda.ml` verbatim**,
  and unlike `__half` — a built-in on both toolchains — the bf16 spellings
  diverge (`__nv_bfloat16` from `cuda_bf16.h` vs ROCm's `__bf16`/`hip_bfloat16`).
  One arm in the shared emitter cannot serve both.
- **Metal** — advertises `bfloat` (MSL 3.1+), unmeasured: no Apple device or
  Metal toolchain here.
- **OpenCL / GLSL-Vulkan / WGSL** — bf16 support is not established. RADV on the
  RX 7900 XTX (Mesa 26.1.4-arch3.1) advertises `VK_KHR_shader_float16_int8` and
  **no bfloat16 extension**, despite RDNA3 having bf16 in its ISA.
- **PTX** — refuses f16 already, for the `%h` register-class reason.

Under the capability model's own rule — **`Unknown` does not permit** — every one
of those is a refusal. A bf16 element type that every backend refuses is not a
feature; it is a constructor that lengthens 38 match sites and delivers nothing.

**And the contraction question is open, not answered.** AMD's two GPU compilers
fuse an f32 multiply into the f32→**f16** narrowing (620/63488 through
LLVM/AMDGPU and ACO; 2912/63488 through SPIR-V, where the add is swallowed too).
Whether the same combine exists for f32→**bf16** is **unmeasured**. It must not
be assumed either way — assuming *yes* would refuse a backend that works, and
assuming *no* would ship the §6.2 discipline broken. That measurement is a
prerequisite for a bf16 backend, and it is not in this change.

### 5.3 What would make bf16 cheap

In order, none of which is bf16 itself:

1. **A host storage decision for widths `Bigarray` does not have** — the real
   §11.3 question, and it is a *type-level* problem, which is why §11.3's
   predicted 60% ceiling for a value-level table is if anything optimistic.
2. **The f32→bf16 narrowing-fusion sweep** on gfx1100 and both RADV devices, the
   same 65536-input exhaustive shape as the f16 sweeps, with the barriered
   control that proves it can go green.
3. **Splitting `Sarek_ir_cuda.ml`'s type mapper for the CUDA/HIP divergence**,
   which f16 never needed because `__half` happened to be spelled the same.

Until (1) is answered, bf16 is not a width addition. It is a storage-layer design
task wearing a width addition's clothes.

---

*Every count in this document was produced by building the tree, not by reading
it. The measurement patch (719 lines) reached a green build with `TBFloat16` and
`BFloat16` present in all four addable vocabularies, and was reverted; only the
sweep in §3 is proposed for merge.*
