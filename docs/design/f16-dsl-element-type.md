# Design Spec: f16 (half-precision float) as a Sarek DSL element type

**Task:** #57 (`f16-dsl-element-type`) — roster RESEARCH + SPEC phase.
**Status:** DESIGN SPEC FOR REVIEW. No implementation. Design forks are documented
inline with a recommendation each; the human reviews before any code is written.
**Date:** 2026-07-25.
**Method:** float32 / float64 traced end-to-end as the template; each layer mapped
to what f16 needs, with `file:line` anchors verified against the worktree.

---

## 0. TL;DR

- **f16 SCALAR is the prerequisite** that unlocks every tensor-core path in Sarek:
  Vulkan cooperative-matrix (#62), Metal `simdgroup_matrix` (#63), HIP rocWMMA from
  `[%kernel]` (#44 étape B). None are reachable until the DSL can express an f16
  element type. This spec covers the f16 **scalar** type; f16 **matrix/fragment**
  types are a delineated follow-on (§8, slice 3).
- **The host storage layer is the EASY part, not the hard part** — contrary to a
  stale note in `docs/optimization/opt-expressivity-gaps.md:82` and one of the
  research passes. **Verified empirically** (OCaml 5.3.0, this switch):
  `Bigarray.float16_elt` and `Bigarray.Float16` both exist, and
  `Array1.create/get/set` round-trip through binary16 for free
  (`3.14159 → 3.14062`). Size knowledge is already wired
  (`sarek/core/Memory.ml:23`, `sarek/core_base/Spoc_core_base_scalar.ml:36`).
- **The genuinely invasive parts are (a) the PTX backend** (register class is
  detected by string-prefix sniffing of register names, not a type match) and
  **(b) the sheer breadth** — two IR representations plus a typer type vocabulary,
  ~40 exhaustive `elttype` matches, and per-plugin codegen copies.
- **Reserved-name groundwork is already done:** `float16`, `half`, `half2..half16`
  are reserved (`sarek/ppx/Sarek_reserved.ml:124-135`); `f16` is a WGSL reserved
  keyword (`sarek/codegen/Sarek_ir_wgsl.ml:53`).
- **Recommended surface:** `float16 vector` type annotation (matches the reserved
  name and the `float32`/`float64` pattern); **no f16 literal** in slice 1
  (only conversions), because the literal-suffix scheme has no natural free suffix.
- **Recommended arithmetic semantics:** **promote-to-f32-and-round** ("storage
  type, compute in f32"), which is what every backend does natively and what keeps
  cross-backend results consistent. Native f16 arithmetic is a slice-3 optimization.
- **Recommended capability gating:** add `supports_fp16` to the device
  capabilities record and reuse the exact `kernel_uses_float64` →
  emit-pragma/extension machinery (a parallel `kernel_uses_float16` detector),
  plus a launch-time gate — the #64 static-where-known + dynamic-gate model.

---

## 1. Why this matters — the tensor-core unlock

The expressivity-gap analysis already ranks these
(`docs/optimization/opt-expressivity-gaps.md:82,86`):

- **#62 Vulkan cooperative-matrix** (`VK_KHR_cooperative_matrix`) — cross-vendor
  tensor path. Cooperative-matrix fragments are typically f16-input / f32-accumulate.
- **#63 Metal `simdgroup_matrix`** — `simdgroup_half8x8` fragments are f16.
- **#44 étape B HIP rocWMMA** — the `mma`-probe (`docs/optimization/l15b-mma-probe.md`)
  shows the WMMA path assembles but is not reachable from `[%kernel]` today; the
  probe kernel itself is `...f32.f16.f16.f32` — f16 inputs.

All three consume an **f16 fragment element type**. You cannot express a
cooperative-matrix / simdgroup-matrix / WMMA fragment in the DSL until the DSL has
an f16 scalar element type to build the fragment type on. Hence f16 scalar is the
foundation; this spec builds it, and §8 slice 3 delineates the matrix layer on top.

Honest scope note carried over from the gap analysis: f16 is **high value for ML
workloads, ~zero relevance for the FP32/FP64 numeric-kernel user base Sarek serves
today** (`opt-expressivity-gaps.md:82`). This spec is written because f16 is a
*structural prerequisite* for the tensor-core roadmap, not because f16 scalar math
is itself in demand. That framing should inform how much of the slice plan is worth
funding now (§8, §9).

---

## 2. The template: how float32 / float64 thread end-to-end

Sarek carries **three** type vocabularies a float width flows through. This is the
central structural fact that sizes the work.

| Layer | Vocabulary | float32 / float64 |
|---|---|---|
| Typer front end | `Sarek_types.registered_type` | `Float32` / `Float64` (`sarek/ppx/Sarek_types.ml:34-40`) |
| Legacy IR (PPX lowering) | `Kirc_Ast.elttype` | `EFloat32` / `EFloat64` (`sarek/ppx/Kirc_Ast.ml:47`) |
| New IR (codegen + interp) | `Sarek_ir_types.elttype` | `TFloat32` / `TFloat64` (`spoc/ir/Sarek_ir_types.ml:49-50`) |
| Host storage | `scalar_kind` GADT | `Float32` / `Float64` (`sarek/core_base/Spoc_core_base_scalar.ml:17-18`) |

Adding f16 means adding a constructor in **each** vocabulary and extending every
exhaustive match over it. The blast radius is enumerated in §5.

---

## 3. Per-layer map (what f16 needs at each layer)

### 3.1 Type system / kind GADTs

**Host `scalar_kind` GADT** — `sarek/core_base/Spoc_core_base_scalar.ml:16-22`:

```ocaml
type (_, _) scalar_kind =
  | Float32 : (float, Bigarray.float32_elt) scalar_kind
  | Float64 : (float, Bigarray.float64_elt) scalar_kind
  | ...
```

f16 needs: `| Float16 : (float, Bigarray.float16_elt) scalar_kind`.

> **VERIFIED (settles a research contradiction).** One research pass claimed stock
> OCaml `Bigarray` has no `float16_elt` and flagged host storage as "the hard part."
> That is stale (was true pre-OCaml-5.2). I compiled and ran a probe in this switch
> (OCaml 5.3.0; `dune-project` pins `>= 5.4.0`): `type t = (float,
> Bigarray.float16_elt) Bigarray.kind`, `Bigarray.Array1.create Bigarray.Float16`,
> `set 3.14159`, `get → 3.14062`. **`float16_elt`, `Float16`, and binary16
> round-on-store all exist natively and unpatched.** `dependencies/` carries no
> bigarray patch. Host storage for f16 is therefore a *mechanical* addition, and the
> binary16 rounding the emulation strategy needs (§6) comes for free from the
> Bigarray store.

Downstream host helpers to extend (all in `Spoc_core_base_scalar.ml`, 1:1 with the
float64 arm): `to_bigarray_kind` (`:26-33` → `Float16 -> Bigarray.Float16`),
`scalar_elem_size` (`:52-58` → `Float16 -> 2`), `scalar_kind_name` (`:60-62`),
`scalar_type_id` (`:88-91`). Re-export the constructor at `Spoc_core_base.ml:83-89`
+ `.mli`. Public `kind` / `host_storage` GADTs (`Spoc_core_base.ml:107-120`) need no
new constructor — f16 rides in as `Scalar Float16` / `Bigarray_storage` like any
scalar.

Convenience constructors mirror `float32`/`float64`: `Spoc_core_base.ml:487-497`,
`Vector.ml:191-205` (`let float16 = Scalar Float16`, `create_float16`).

**No `kind` GADT in `spoc/` proper** — the host element-type GADT lives in
`sarek/core_base/`. The `spoc/` layer is byte-oriented (see §3.3).

### 3.2 IR element type

**New IR** — `spoc/ir/Sarek_ir_types.ml`:
- `type elttype` (`:46-59`): add `| TFloat16`.
- `type const` (`:70-76`): add `| CFloat16 of float` **iff** literals are supported
  (fork §6.1 recommends deferring literals → `CFloat16` optional in slice 1).

**Legacy IR** — `sarek/ppx/Kirc_Ast.ml:47` (`elttype = EInt32|EInt64|EFloat32|EFloat64`)
and node region `:244-249`: add `EFloat16`. The PPX lowers to this rep in
`Sarek_lower.ml`; the transpiler (`sarek/transpile/Sarek_ir_conv.ml:104`) bridges
legacy↔new, so both need the arm.

### 3.3 Host storage / transfer

**Transfer is element-type-agnostic beyond byte size — no change needed.**
`sarek/core/Memory.ml`: `host_to_device` (`:162-168`) and `device_to_host`
(`:173-179`) compute `byte_size = dim * elem_size` and move raw pointers; the
`BUFFER` signature (`:45-69`) transfers via `unit Ctypes.ptr` + `~byte_size:int`
only (comment `:42-44`: "raw pointers with byte sizes"). `alloc` (`:78-114`) already
derives `elem_size = bigarray_elem_size kind`, which returns 2 for `Float16`
(`:23`). A 2-byte `Float16` bigarray transfers correctly by byte count with zero
transfer-layer changes. Whether the device interprets the bytes as IEEE binary16 is
a codegen concern, not a transfer concern.

Vector get/set are storage-generic (`Vector.ml:74-137`): `Bigarray.Array1.get/set`
on a `Float16` array already returns/accepts an OCaml `float` and rounds on store —
the same free round-trip verified in §3.1.

**Net:** host storage + transfer for f16 = add the `scalar_kind` constructor + its
~4 helper arms + convenience constructors. That is all.

### 3.4 Typer

`sarek/ppx/Sarek_typer.ml` uses `Sarek_types.typ`, not the IR elttype:
- `registered_type` (`Sarek_types.ml:34-40`): add `Float16`.
- Literal inference (`Sarek_typer.ml:235-246`): `EFloat` is polymorphic and defaults
  to float32; `EDouble` → `t_float64`. f16 participation in the float-literal
  defaulting lattice (`Sarek_types.ml:331-335`, guard `:150` whitelists
  `Float32|Float64`) must be decided — see fork §6.1.
- Numeric predicates enumerating float widths: `Sarek_typer.ml:97`,
  `Sarek_types.ml:150,353`, pretty-printer `:242-243`.
- **typ → IR elttype** happens in lowering, not the typer:
  `Sarek_lower_ir.ml:22-32` (`TReg Float32 -> Ir.TFloat32`) for the new IR;
  `Sarek_lower.ml:216-220` for the legacy IR + per-width node choices
  (`:434-436,504-506,639-641,707-715`).

**Type annotation resolution** (how a user writes the type): `Sarek_lower.ml:128-139`
`Ptyp_constr {txt=Lident "float32"} -> TReg Float32`. Add `"float16" -> TReg Float16`
(and optionally `"half"` as an alias — fork §6.1).

### 3.5 Conversions / casts

Two distinct mechanisms:
- **PPX front end:** width conversions are **intrinsics emitting device cast
  strings**, not a syntactic cast. `Float32.of_int` → `"(float)(%s)"`
  (`sarek/Sarek_stdlib/Float32.ml:166-170`); `Float64.float64` emits `"double"`
  (`sarek/Sarek_float64/Float64.ml:23-24`). Legacy AST has a dedicated
  `CastDoubleVar` node (`Kirc_Ast.ml:70`). An f16 surface needs a parallel
  `Float16` intrinsic module (`f16_of_float` / `float_of_f16`).
- **IR level:** `Sarek_ir.ECast of elttype * expr`
  (`spoc/ir/Sarek_ir_types.ml:121`) carries the target elttype; handled in
  `Sarek_ir_conv.ml:104`, interp `Sarek_ir_interp_eval.ml:220-226`, PTX
  `Sarek_ir_ptx_expr.ml:530` + `Sarek_ir_ptx_types.ml:175-187`, and float64 analysis
  `Sarek_ir_analysis.ml:181`. Each needs a `TFloat16` arm and f16↔f32↔f64 promotion
  rules (currently a binary "is float64" question becomes 3-way — see §6.2).

### 3.6 Native codegen + Interpreter

**f64 is NOT emulated on native/interp — it is just OCaml `float`.** The interpreter
represents both widths as `PFloat of float`
(`sarek/interp/.../Typed_value.ml:27`); `Float32_type` and `Float64_type` have
identical `type t = float` (`:216-241`); intrinsic dispatch distinguishes them only
by module path (`Sarek_ir_interp_value.ml:314-328`). So there is no f64 "emulation"
to copy.

The correct f16 native/interp template is: represent f16 as OCaml `float` (a
`Float16_type` sharing `PFloat`), and get binary16 rounding for free from the
`Bigarray.Float16` store on write-back (§3.1). Value variant `VFloat32`/`VFloat64`
(`sarek/interp/Sarek_value.ml:17-18,29-30`) gains `VFloat16`; zero-values / cast eval
/ defaults (`Sarek_ir_interp_eval.ml:105-106,220-226,377-378`) gain the arm.

> **Clarification on "softmath".** `sarek/codegen/Sarek_ir_softmath.ml` is a **GPU
> backend** facility (software f64 transcendentals for PTX and GLSL, which lack f64
> transcendental instructions), dispatched at `Sarek_ir_analysis.ml:331`. It is
> **not** the native/interp emulation template. It becomes relevant only if f16 GPU
> *transcendentals* are needed on a backend lacking native f16 math — a slice-2/3
> concern, not slice 1.

**Native execution bridge** — `native_arg` (`Sarek_ir_types.ml:185-211`):
`NA_Float32`/`NA_Float64` both carry plain `float`. Producers
(`sarek/sarek/Kirc_kernel.ml:99-269`, `sarek/plugins/native/Native_plugin.ml:167-277`)
and PPX-generated consumers (`sarek/ppx/Sarek_native_gen_kernel.ml:32-90,183-191`)
map element widths to `get_f32`/`get_f64` accessors. See fork §6.4 for whether f16
reuses `get_f32`+rounding or adds `NA_Float16`/`get_f16`.

---

## 4. Per-backend codegen map

Each backend has a type-string function, a constant emitter, and (for some) a
feature/extension declaration. All verified.

| Backend | file (type fn) | f32 / f64 today | f16 type string | f16 literal | feature declaration for f16 |
|---|---|---|---|---|---|
| **CUDA / HIP** | `sarek/codegen/Sarek_ir_cuda.ml:46` | `"float"` / `"double"` (`:49-50`) | `half` (or `__half`) | none — `__float2half(x)` | add `#include <cuda_fp16.h>` to `cuda_header` (`:656`) |
| **OpenCL** | `Sarek_ir_opencl.ml:57` | `"float"` / `"double"` (`:60-61`) | `half` | suffix `h` (`1.5h`) | prepend `#pragma OPENCL EXTENSION cl_khr_fp16 : enable` — mirror `generate_with_fp64` (`:755-759`) |
| **GLSL / Vulkan** | `Sarek_ir_glsl.ml:226` | `"float"` / `"double"` (`:229-230`) | `float16_t` | suffix `hf` | add `uses_float16` to `glsl_header` (`:1078`) → `#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require` (+ likely `GL_EXT_shader_16bit_storage` for buffer I/O) |
| **WGSL** | `Sarek_ir_wgsl.ml:133` | `"f32"` / **rejects f64** (`:141-143`) | `f16` (native) | suffix `h` | prepend `enable f16;` to **module top** (before bindings, not in `wgsl_header`); do NOT reject in `params_have_float64` (`:1171`) |
| **Metal** | `Sarek_ir_metal.ml:46` | `"float"` / `"float"` (f64→float, `:49-50`) | `half` (native) | suffix `h` | none — native in `metal_stdlib` (already `#include`d `:909`) |
| **PTX** | `Sarek_ir_ptx_types.ml:172` | `.f32` / `.f64` (`:175-176`) | `.f16`, new reg class `%h` | `mov.f16 %r, 0H%04X;` (bit pattern) | none — default `sm_86` ≥ sm_53 |

**Constant emission template** (the `^ "f"` idiom): CUDA `Sarek_ir_cuda.ml:93,100`,
OpenCL `:109,116`, GLSL `:333,353` (double uses `"lf"`), Metal `:113,120`, WGSL
`:242`, PTX `Sarek_ir_ptx_expr.ml:436,440` (hex bit-pattern). f16 literals (if
adopted, §6.1) mirror these with the per-backend suffix column above.

**Feature-declaration mechanism (the #64 tie-in).** The OpenCL and GLSL paths are the
model: a `Sarek_ir_analysis.kernel_uses_float64 k` predicate gates a prepended
pragma / `#extension` line. f16 adds a **parallel `kernel_uses_float16`** detector
(§3.2 / §5) and reuses the identical prepend structure. See §7.

**Two backends diverge from the float64 template and carry real risk:**
- **WGSL** currently *rejects* f64 (`:141-143`, `:248`). f16 is the opposite
  direction — a genuinely new *supported* path — so there is no reject-arm to copy;
  and `enable f16;` must land at the module top, before the bindings the header
  emits (`:1198,1233`).
- **PTX** identifies register class by **string-prefix sniffing of register names**
  (`emit_cast` `Sarek_ir_ptx_expr.ml:1177-1180`: `is_f64` = name starts `%fd`,
  `is_f32` = `%f` and not `%fd`). A new `%h` (f16) class means auditing and
  extending every prefix-based guard (`emit_cast`, `emit_bitwise` `:1166`), plus
  `cvt.rn.f16.f32` / `cvt.f32.f16` in every float arm — not a mechanical match-arm
  addition. This is the highest-risk single file.

**Per-plugin duplication.** `sarek-cuda/`, `sarek-opencl/`, `sarek-hip/`,
`sarek-metal/`, `sarek-vulkan/` each ship a `Sarek_ir_<backend>.ml` that mostly
`include`s the shared codegen but must be audited for local elttype matches. HIP
reuses CUDA codegen verbatim (`sarek-hip/Hip_plugin.ml:36`, `Hip_shared.ml:12-13`).

---

## 5. Blast radius — exhaustive matches that adding a constructor forces

Adding `TFloat16` / `EFloat16` / `Float16` / `VFloat16` turns every exhaustive
`elttype` match into a compile error until extended. This is the checklist (the
compiler is the driver — non-exhaustive-match warnings enumerate the rest). ~40
sites, grouped:

- **IR defs:** `Sarek_ir_types.ml:46-59,70-76`; `Kirc_Ast.ml:47,244-249`.
- **spoc/ir core:** `Sarek_ir_layout.ml:97-98,110-111,126,185` (size/align);
  `Sarek_ir_pp.ml:13-14`; `Sarek_ir_analysis.ml:165,173,181` (add parallel
  `*_uses_float16` chain: `elttype_uses_float16`, `const_uses_float16`,
  `float16_leaf`, `float16_folder`, `kernel_uses_float16`); `Sarek_pure_registry.ml`.
- **PPX pipeline:** `Sarek_lower.ml:26-27,128-139,216-220,306-308,434-436,504-506,639-641,707-715`;
  `Sarek_lower_ir.ml:22-32`; `Sarek_types.ml:34-40,150,242-243,353`;
  `Sarek_typer.ml:97,235-246`; `Sarek_parse.ml:135-138`;
  `Sarek_quote_ir.ml:176-178`, `Sarek_quote.ml:154-156`; plus audit
  `Sarek_env.ml`, `Sarek_mono.ml`, `Sarek_core_primitives.ml`,
  `Sarek_native_gen*.ml`, `Sarek_native_intrinsics.ml`.
- **transpile / interp:** `Sarek_ir_conv.ml:29-30,104`;
  `Sarek_ir_interp_eval.ml:105-106,220-226,377-378`; `Sarek_value.ml:17-18,29-30`;
  `Typed_value.ml:216-241`; `Sarek_ir_interp_value.ml:314-328`.
- **codegen backends:** the §4 table sites + `Sarek_ir_ptx_*` register class /
  cvt / mem; `Sarek_ir_softmath.ml` (only if f16 GPU transcendentals needed);
  `Sarek_ir_intrinsic_dispatch.ml`.
- **host runtime:** `Spoc_core_base_scalar.ml` (~6 matches, §3.1) +
  `Spoc_core_base.ml`/`.mli`; `Vector.ml:228,306,316`; `Kernel_arg.ml:51,77`;
  `Execute.ml:102+`; `Soa.ml:26`, `Sarek_tuple_vec.ml`;
  `core_js_exec/Kernel_arg_jsx.ml`.
- **plugins:** each `sarek-*/Sarek_ir_<backend>.ml` (§4).

---

## 6. Design forks — recommendations

Each fork states a recommendation + rationale. Forks flagged **⚠ WANT HUMAN INPUT**
are where I would not proceed to implementation without a decision.

### 6.1 DSL surface: how does a user declare/annotate f16?

**Options:** (a) `float16 vector` type annotation; (b) a `half` alias; (c) an f16
literal syntax.

- **Recommendation:** primary surface is **`float16 vector` / `(x : float16)`**
  annotation, matching the `float32`/`float64` pattern and the already-reserved
  `float16` name. Add `half` as a **thin alias** resolving to the same `TReg
  Float16` (both names are already reserved) — cheap, and `half` is the idiom GPU
  programmers expect.
- **No f16 literal in slice 1** (recommendation). The literal-width scheme is
  suffix-driven: bare = float32, `G`/`g` = float64 (`Sarek_parse.ml:135-138`).
  OCaml float literals allow a single-char suffix; `h`/`H` is unused and would read
  as "half", but: (i) f16 literals are rarely needed — f16 is a storage/interchange
  type, values are almost always produced by conversion or computed in f32; (ii)
  adding a literal pulls in `CFloat16` const + typer defaulting-lattice changes for
  little benefit. Defer literals; users write `f16_of_float 1.5` (see §6.3). If a
  literal is later wanted, `1.5h`/`1.5H` is the natural choice.

  **⚠ WANT HUMAN INPUT** on whether the `half` alias is desired or whether
  `float16` alone is cleaner (one name, less surface). Low stakes, but it is a
  public-API taste call.

### 6.2 Arithmetic semantics: compute in f16 or promote to f32?

**Recommendation: "storage type, compute in f32 and round"** — f16 values are
stored/loaded as binary16; arithmetic promotes to f32, computes, and the result is
rounded back to f16 on store. Rationale:

- It is what the hardware does anyway on most targets for scalar (non-packed) f16
  ops, so it is not a pessimization for the common case.
- It is the **only** way to keep cross-backend results consistent, which is Sarek's
  core value prop. Native per-lane f16 rounding differs subtly across
  CUDA/Metal/Vulkan/WGSL; forcing f32 compute + a single defined round-to-binary16
  makes the interpreter, native, and every GPU backend agree — exactly the
  consistency discipline `Sarek_real64` already applies to f64
  (`sarek/Sarek_real64/Sarek_real64.ml`), where the op set is the *intersection* of
  substrates and results are defined to match.
- On native/interp it is free: compute in OCaml `float`, round on write-back through
  the `Bigarray.Float16` store (§3.1, verified).

**Deferred to slice 3:** true native f16 arithmetic and **packed math** (`half2` /
`__half2` / `f16x2`, PTX `hfma2`). Packed math is where f16's throughput advantage
actually lives, and it is the tensor-core adjacency — but it is an *optimization on
top of* correct scalar f16, not a prerequisite.

### 6.3 Conversions: explicit intrinsics vs implicit

**Recommendation: explicit intrinsics, no implicit f16↔f32 coercion.** Provide a
`Float16` stdlib module with `f16_of_float : float32 -> float16` and `float_of_f16 :
float16 -> float32` (plus `of_int`), mirroring `Float32`/`Float64`
(`Sarek_stdlib/Float32.ml:166-170`, `Sarek_float64/Float64.ml:23-24`). Lower each to
`Sarek_ir.ECast (TFloat16, _)` / `ECast (TFloat32, _)`, then per-backend to the
native convert (CUDA `__float2half`/`__half2float`, PTX `cvt.rn.f16.f32`/`cvt.f32.f16`,
GLSL `float16_t(x)`/`float(x)`, WGSL `f16(x)`/`f32(x)`, Metal `half(x)`/`float(x)`).

Rationale: matches how float32↔float64 already works (explicit, intrinsic-based, no
silent widening in device code); avoids surprising precision loss from implicit
narrowing; keeps the typer's float-defaulting lattice simple (f16 stays outside the
bare-literal defaulting set, §6.1).

### 6.4 Native-arg representation: reuse `get_f32` or add `NA_Float16`?

**Recommendation: reuse `get_f32`/`set_f32` with rounding for slice 1.** Because the
native/interp layer already represents every float as OCaml `float` and
`NA_Float32`/`NA_Float64` are byte-identical `of float`, an f16 vector's set is "write
a float into a `Bigarray.Float16` cell" — rounding is a property of the *storage
kind*, not the accessor. No new `get_f16`/`set_f16` field is strictly required.

Add `NA_Float16` + `get_f16`/`set_f16` **only if** f16 *scalar-by-value* args must
enforce binary16 rounding (no bigarray to round through), or the interpreter must
distinguish f16 scalars for intrinsic dispatch. Recommend adding `NA_Float16 of
float` with a round-to-binary16 on construction **if** scalar f16 kernel params are
in scope for slice 1; otherwise defer. **⚠ minor** — depends on whether f16 scalar
params (not just vectors) are wanted early.

### 6.5 Capability gating — see §7 (it is the #64 tie-in, treated as its own section).

---

## 7. Capability gating (#64 tie-in)

f16 support is **not universal** (many OpenCL/Vulkan devices lack it; WebGPU requires
the `shader-f16` feature; older PTX targets pre-sm_53). An f16 kernel on a
backend/device without f16 must be caught. The existing f64 machinery is the exact
template.

**Static, where known (compile/lower time):**
- WGSL already rejects f64 statically (`Sarek_ir_wgsl.ml:141-143`) via
  `params_have_float64`/`has_float64`. The same shape gives a clear diagnostic when
  a backend that structurally cannot do f16 is targeted.
- The `kernel_uses_float64` detector (`Sarek_ir_analysis.ml:205`) drives conditional
  emission of `cl_khr_fp64` / `GL_ARB_gpu_shader_fp64`. **Add a parallel
  `kernel_uses_float16`** (same fold, over `CFloat16`/`TFloat16` leaves) that drives
  the f16 pragma/extension/`enable` prepends in §4.

**Dynamic, at launch (device-dependent):**
- The device capabilities record (`spoc/framework/Framework_sig.ml:26-41`) has
  `supports_fp64` and `supports_atomics` but **no `supports_fp16`**. Add
  `supports_fp16 : bool`, populated per backend probe (CUDA compute-capability ≥
  5.3; OpenCL `cl_khr_fp16` query; Vulkan `shaderFloat16` feature; WebGPU
  `shader-f16` feature; Metal always true on supported GPUs).
- Add `Device.allows_fp16 d = d.capabilities.supports_fp16` and a `with_fp16 ()`
  filter, mirroring `allows_fp64`/`with_fp64` (`sarek/core/Device.ml:129-132,186`).
- Launch gate: when a kernel where `kernel_uses_float16` is true is run on a device
  where `allows_fp16` is false, raise a clear runtime error (mirror the fp64 path).

**Substrate-selection analogy (informative, likely NOT needed for f16).**
`Sarek_real64` picks `Native_f64` vs `Fallback_df64` from `Device.allows_fp64`
(`Sarek_real64.ml:114-122`) and authors a kernel once, lowering to both. For f16 the
"fallback when the device lacks f16" is trivially *compute in f32* — most devices do
this natively — so a full df64-style dual-lowering is likely unnecessary. The
recommendation (§6.2) of "compute in f32, round to f16" means a device lacking
*storage* f16 is the only real gap, and that is rare. Keep the launch gate; skip a
df64-style substrate machine unless a concrete no-f16-storage device appears.

**⚠ WANT HUMAN INPUT:** whether the launch-time gate should be a hard error or a
warn-and-emulate-in-f32 fallback. My recommendation is **hard error** (Sarek's
proof-tier discipline favors explicit over silent), but this composes with #64's
policy and is the kind of call that should be made deliberately.

---

## 8. Slice plan

f16 **scalar** is the prerequisite; f16 **matrix/fragment** types are the follow-on.
Slices are independently reviewable and each ends at a green build + tests.

### Slice 1 — foundation: host storage + type system + Native/interp + one GPU backend
**Goal:** `float16 vector` is a real element type end-to-end on the interpreter,
native, and **one** GPU backend, with compute-in-f32 semantics and explicit
conversions.
- Host: `Float16` scalar_kind + helpers + `create_float16` (§3.1). *Low risk —
  verified Bigarray support.*
- Type system: `registered_type Float16`, annotation resolution `"float16"`(+`half`),
  typer numeric predicates (§3.4). No literal (§6.1).
- Both IR reps: `TFloat16` / `EFloat16`; `Sarek_ir_conv` bridge; `ECast` handling;
  `Float16` intrinsic module with `f16_of_float`/`float_of_f16` (§6.3).
- Native/interp: `VFloat16`/`Float16_type` sharing `PFloat`; reuse `get_f32`+rounding
  (§6.4).
- **One GPU backend: recommend CUDA/HIP** (simplest feature decl — a single
  `#include`; and it is the tensor-core target for #44). `TFloat16 -> "half"`.
- Capability: `supports_fp16` field + `kernel_uses_float16` detector + launch gate
  (§7).
- **Deliberately excludes PTX** (see slice 2 rationale).

### Slice 2 — remaining backends
- OpenCL, GLSL/Vulkan, WGSL, Metal type strings + literal suffixes + feature
  declarations (§4 table). Each is mechanical *except* WGSL (new supported path,
  module-top `enable f16;`).
- **PTX** — carved out here because of the string-prefix register-class sniffing
  risk (§4). Treat as its own reviewable unit: new `%h` register class, `is_f16`
  guards in `emit_cast`/`emit_bitwise`, `cvt.*.f16.*` conversions, `mov.f16` const.
- Per-plugin `Sarek_ir_<backend>.ml` audit (§4).

### Slice 3 — f16 matrix / tensor-core types (the unlock; separate spec)
**This slice is a follow-on and deserves its own design spec.** It is where #62/#63/#44
are actually delivered:
- Packed f16 math (`half2`/`f16x2`, `hfma2`) — the throughput win.
- An f16 **fragment/matrix** element type in the IR (a new type class, not just a
  scalar), plus per-backend fragment ABI: PTX `mma.sync`/`wmma`, Metal
  `simdgroup_matrix`, Vulkan `VK_KHR_cooperative_matrix`. OpenCL has no portable
  equivalent (`opt-expressivity-gaps.md:86`).
- Gated by the same `supports_fp16` + a new `supports_cooperative_matrix`-class
  capability.

Slice 3 is explicitly **out of scope** for #57; #57 delivers slices 1–2 (f16 scalar).
The matrix layer is named here only to fix the relationship.

---

## 9. Honest effort / risk per slice

| Slice | Effort | Risk | Notes |
|---|---|---|---|
| 1 (foundation + CUDA) | **M** | **Low–Med** | Host storage verified-easy. Breadth is the cost, not depth: ~25 mechanical match-arm edits across two IR reps + typer + host + one backend, driven by the compiler. Main design risk is the typer float-defaulting lattice interaction (§3.4) — contained by "no f16 literal". |
| 2 (other backends) | **M** | **Med** | OpenCL/GLSL/Metal/WGSL mechanical. **PTX is the real risk** — string-prefix register-class detection is fragile and untyped; a `%h` class touches guards that were never written to be extended. Budget PTX as its own focused effort with the `test_ptx_mma_probe.ml`-style hand-PTX validation. |
| 3 (matrix/tensor-core) | **XL** | **High** | New IR type class + 3 divergent backend fragment ABIs + no OpenCL path. Own spec. This is where the `opt-expressivity-gaps.md:86` "XL / doc-only" verdict lives — the value is real but it fragments the "one kernel, six backends" prop. |

**Cross-cutting risks / honesty:**
- **Duplication tax:** two IR representations (`Kirc_Ast` legacy + `Sarek_ir` new)
  and per-plugin codegen copies mean every change is made ~twice. If a
  legacy-IR-removal is on any roadmap, doing that *first* would roughly halve slice
  1–2 cost. **⚠ worth a human decision:** is the legacy `Kirc_Ast` path still
  load-bearing, or can f16 target only the new IR?
- **Demand mismatch:** f16 scalar math has ~zero current user demand
  (`opt-expressivity-gaps.md:82`); the entire justification is the slice-3 unlock.
  If slice 3 is not going to be funded, slices 1–2 buy little on their own beyond
  storage/interchange. This is the most important strategic question and it is a
  human call, not an engineering one.
- **Verification substrate:** ZLUDA does not execute `mma`/`wmma`
  (`l15b-mma-probe.md`), so slice-3 tensor-core validation needs real
  NVIDIA/AMD/Apple hardware, not the current dev box. Slices 1–2 are validatable on
  the interpreter + one real GPU.

---

## 10. Open questions I would want answered before implementing

1. **Legacy IR:** is `Kirc_Ast` still required, or may f16 target only `Sarek_ir`?
   (Halves the blast radius if the latter.)
2. **Strategic:** is slice 3 (tensor cores) funded? If not, is f16 storage/interchange
   alone worth slices 1–2? (§9.)
3. **Launch gate policy:** hard error vs warn-and-emulate on an f16-lacking device
   (§7). Composes with #64.
4. **Surface taste:** `half` alias alongside `float16`, or `float16` only? (§6.1.)
5. **f16 scalar params:** are by-value f16 kernel params in scope for slice 1
   (→ `NA_Float16`), or vectors-only first? (§6.4.)

---

*Anchors verified against the worktree at commit-time. The `float16_elt` /
`Bigarray.Float16` availability claim in §0/§3.1 was verified by compiling and
running a probe in this OCaml switch, not assumed.*
