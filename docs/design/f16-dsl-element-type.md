# Design Spec: f16 (half-precision float) as a Sarek DSL element type

**Task:** #57 (`f16-dsl-element-type`) — roster RESEARCH + SPEC phase.
**Status:** DESIGN SPEC, **AMENDED POST-IMPLEMENTATION**. The original text was
written before any code existed; slice 1 was then built (PR #290) and proved parts
of this spec wrong. Those parts are corrected **in place**, next to the claim they
refute, and flagged with a `CORRECTION` callout. The spec is deliberately kept as a
design document rather than rewritten into an implementation report — the point of
keeping the wrong claims visible is that the *next* width's spec gets written
better.
**Date:** written 2026-07-25; amended 2026-07-25 (post PR #290 / #291).
**Method:** float32 / float64 traced end-to-end as the template; each layer mapped
to what f16 needs, with `file:line` anchors verified against the worktree.

**Reading the callouts:**

- **`CORRECTION`** — the spec asserted X, implementation proved Y. Every correction
  in this document is **VERIFIED BY EXECUTION** (a compiled probe, a run kernel, a
  compiler diagnostic) unless it explicitly says otherwise.
- **`HELD`** — a design call the implementation exercised and did not have to revisit.
- Unmarked prose is the original, reasoned, pre-implementation text. Treat it as
  reasoned-not-verified unless a callout upgrades it.

**Anchor hygiene.** All `file:line` anchors in the original text were verified
against the worktree at writing time. They have since drifted: PR #77 factorized
the C-family codegen, PR #78 (`ba375a36`) deleted `Kirc_Ast.ml` and
`Sarek_lower.ml` outright, and PR #290 itself moved things. Anchors re-verified
during this amendment carry their current line; anchors that could not be
re-verified are marked *(approx.)* rather than guessed at.

---

## Post-implementation corrections — read first

Slice 1 shipped as **PR #290** (`feat/f16-dsl-slice1`, three commits: initial
implementation, review-round-1 must-fixes, review-round-2 duplication collapse +
soundness holes). **PR #291 / issue #78** (`ba375a36`) separately deleted the dead
legacy `Kirc_Ast` path (~2,215 lines removed). Together they change five things in
this spec:

1. **§3.3 "transfer needs no change" was WRONG.** `Bigarray.Float16` exists
   (OCaml 5.3, verified) but **Ctypes 0.24.0 has no `Float16` bigarray kind**. Both
   `typ_of_bigarray_kind Float16` and `bigarray_start Float16` raise
   `Failure "Unsupported bigarray kind"`. The spec conflated the Bigarray layer
   (fine) with the Ctypes layer (not fine). See the CORRECTION in §3.3.
2. **§4's `#include <cuda_fp16.h>` was wrong twice.** It would break every HIP
   kernel, and it did not work on CUDA either. See the CORRECTION in §4.
3. **§6.3's `%sarek_intrinsic Float16` module is not buildable as specified** —
   `%sarek_intrinsic` type registration requires a `ctype`, and Ctypes has no half.
   See the CORRECTION in §6.3.
4. **The cost model was wrong in both directions.** The legacy `Kirc_Ast` vocabulary
   was assumed to be a significant duplication tax; it cost exactly one line, and it
   no longer exists. The *real* tax is the `Sarek_ir_types` ⇄ `Sarek_ir_ppx` mirror,
   which turns out to have no dependency justification. See §9.
5. **§7's capability gating was not built in slice 1** and the shape it recommended
   is not the shape slice 1 wanted. See the CORRECTION in §7.

**What held up** is recorded just as explicitly, because it is the half of the spec
a future reader should copy: the compute-in-f32 semantics (§6.2), explicit
conversions (§6.3), the slice plan (§8), the type-system exclusion of f16 from
`is_numeric`/`is_float` (§3.4), and the reserved-name groundwork (§0).

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
  (`sarek/core/Memory.ml:22-23`, `sarek/core_base/Spoc_core_base_scalar.ml:35-36`).

  > **CORRECTION (verified by execution, PR #290 MF2).** Half right, and the wrong
  > half cost the most implementation time in slice 1. *Bigarray* is indeed free.
  > **Ctypes is not**, and the whole host→device path goes through Ctypes. Against
  > ctypes 0.24.0, both `typ_of_bigarray_kind Float16` and `bigarray_start Float16`
  > raise `Failure "Unsupported bigarray kind"` — ctypes' own kind GADT has no
  > `Float16` arm. Every backend's allocation and transfer had to be rerouted
  > through new kind-independent helpers. See §3.3. "Host storage is easy" should
  > have read "the *Bigarray* layer is easy; audit every FFI layer separately."

- **The genuinely invasive parts are (a) the PTX backend** (register class is
  detected by string-prefix sniffing of register names, not a type match) and
  **(b) the sheer breadth** — two IR representations plus a typer type vocabulary,
  ~40 exhaustive `elttype` matches, and per-plugin codegen copies.

  > **CORRECTION (measured, PR #290).** (a) HELD, and was so clearly the risk that
  > slice 1 shipped with PTX rejecting f16 outright via a shared
  > `Sarek_ir_codegen.reject_feature`. (b) Overstated: the measured count is
  > **~20** compiler-forced exhaustive match sites, not ~40, and the "two IR
  > representations" framing put the tax in the wrong place — see §9. The real
  > breadth cost was not in the match arms at all; it was the **FFI layer** (five
  > backends × alloc + transfer, including Vulkan's staging paths and `Metal_api`)
  > and a missing `Float16` arm in `Execute.ml`'s exec-arg `set`
  > (`sarek/execute/Execute.ml:128` on mainline) without which f16 could not run on
  > the **Interpreter device** at all.
- **Reserved-name groundwork is already done:** `float16`, `half`, `half2..half16`
  are reserved (`sarek/ppx/Sarek_reserved.ml:124-135`); `f16` is a WGSL reserved
  keyword (`sarek/codegen/Sarek_ir_wgsl.ml:53`). **HELD** — both anchors still
  exact, and having the names reserved in advance meant the surface decision (§6.1)
  cost nothing to make *or* to leave open.
- **Recommended surface:** `float16 vector` type annotation (matches the reserved
  name and the `float32`/`float64` pattern); **no f16 literal** in slice 1
  (only conversions), because the literal-suffix scheme has no natural free suffix.
  **HELD.** Delivered exactly so (`Sarek_types.ml:458`, `TEConstr ("float16", [])`).
  The `half` alias of §6.1 was **not** taken — `float16` only.
- **Recommended arithmetic semantics:** **promote-to-f32-and-round** ("storage
  type, compute in f32"), which is what every backend does natively and what keeps
  cross-backend results consistent. Native f16 arithmetic is a slice-3 optimization.
  **HELD** — and strengthened: f16 was excluded from `is_numeric`/`is_float`
  entirely, so "compute in f32" is a *type-system guarantee* rather than a
  convention (§3.4).
- **Recommended capability gating:** add `supports_fp16` to the device
  capabilities record and reuse the exact `kernel_uses_float64` →
  emit-pragma/extension machinery (a parallel `kernel_uses_float16` detector),
  plus a launch-time gate — the #64 static-where-known + dynamic-gate model.
  **Half held** — the `kernel_uses_float16` detector shipped, generalized into a
  `feature` type; the `supports_fp16` capability field and launch gate did not. See
  the CORRECTION in §7.

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
| Typer front end | `Sarek_types.registered_type` | `Float32` / `Float64` (`sarek/ppx/Sarek_types.ml:34-40`, still exact) |
| Legacy IR (PPX lowering) | `Kirc_Ast.elttype` | `EFloat32` / `EFloat64` (`sarek/ppx/Kirc_Ast.ml:47`) — **file deleted, see below** |
| New IR (codegen + interp) | `Sarek_ir_types.elttype` | `TFloat32` / `TFloat64` (`spoc/ir/Sarek_ir_types.ml:49-50`, still exact) |
| Host storage | `scalar_kind` GADT | `Float32` / `Float64` (`sarek/core_base/Spoc_core_base_scalar.ml:17-18`, still exact) |

Adding f16 means adding a constructor in **each** vocabulary and extending every
exhaustive match over it. The blast radius is enumerated in §5.

> **CORRECTION (verified, PR #290 + #291/#78).** "Three vocabularies" was the
> central structural claim of this spec and it framed the cost wrongly in both
> directions.
>
> **The legacy `Kirc_Ast` row cost essentially nothing, and no longer exists.**
> `Kirc_Ast.elttype` had **4 constructors and 4 use sites** in the whole tree
> (`Kirc_Ast.ml:47`, `:248-249`, `Sarek_lower.ml:219-220`, `Sarek_quote.ml:51-52`),
> of which a new width forces exactly **one**: `lower_reg_elttype`. Slice 1 did not
> even add a constructor — it added a twelve-line *rejection* arm, because the path
> was already dead. **PR #291 (`ba375a36`, issue #78) then deleted `Kirc_Ast.ml` and
> `Sarek_lower.ml` outright** (~2,215 lines). The legacy IR is no longer a
> consideration for any future width, and §10 question 1 is answered.
>
> **The real duplication tax is a fourth vocabulary this table omits**:
> `spoc/ir/Sarek_ir_types.ml` (runtime) is mirrored by
> `sarek/ppx/Sarek_ir_ppx.ml` (compile-time), bridged by
> `sarek/transpile/Sarek_ir_conv.ml`. That is the pair a future width actually pays
> twice, and §9 records why the mirror is not load-bearing.

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
>
> **CORRECTION (scoped, not refuted).** Everything in this box is true and was
> confirmed by the implementation: the GADT arm went in unchanged at
> `Spoc_core_base_scalar.ml:17` and the binary16 round-on-store is free. What the
> box *implies* — that host storage is therefore done — is what was wrong. The
> Bigarray layer is one of three host layers f16 crosses; the Ctypes FFI layer
> (§3.3) and the `Execute.ml` exec-arg dispatch were both blocking. **Lesson for
> the next width: a probe that proves a layer works is not a probe that proves the
> *path* works. Probe the whole path, or scope the claim to the layer probed.**

Downstream host helpers to extend (all in `Spoc_core_base_scalar.ml`, 1:1 with the
float64 arm): `to_bigarray_kind` (`:26-33` → `Float16 -> Bigarray.Float16`),
`scalar_elem_size` (`:52-58` → `Float16 -> 2`), `scalar_kind_name` (`:60-66`),
`scalar_type_id` (`:88-96`). Re-export the constructor at `Spoc_core_base.ml:83-89`
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

> **CORRECTION (verified).** Do not do this, and it is now impossible to. Slice 1
> added a *rejection* instead of a constructor, on the grounds that the `sarek`
> runtime library exposes no `Kirc_Ast` module, so the `Sarek.Kirc_Ast.*` code this
> path emits does not compile and no PPX artifact references it. **PR #291 then
> deleted the file.** The bridge `Sarek_ir_conv.ml` survived (it bridges
> `Sarek_ir_ppx` ↔ `Sarek_ir_types`, *not* legacy↔new — the spec mis-stated what it
> bridges).

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

> ### CORRECTION — §3.3 is the spec's largest factual error
>
> **VERIFIED BY EXECUTION** (PR #290, review-round-1 finding MF2). "That is all" was
> wrong, and the reasoning behind it contained a specific, nameable conflation worth
> carrying forward.
>
> **What the spec got right:** the *transfer arithmetic* really is element-agnostic.
> `byte_size = dim * elem_size` and the `BUFFER` signature's `unit Ctypes.ptr` +
> `~byte_size:int` are as described, and none of that had to change.
>
> **What it missed:** *acquiring* the pointer is not element-agnostic, because it
> goes through Ctypes, and **Ctypes 0.24.0 has no `Float16` bigarray kind**. Both
> of the following raise `Failure "Unsupported bigarray kind"` (executed probe, this
> switch):
>
> ```
> Ctypes.typ_of_bigarray_kind Bigarray.Float16
> Ctypes.bigarray_start Ctypes.array1 (ba : (float, Bigarray.float16_elt, _) Array1.t)
> ```
>
> The spec conflated **the Bigarray layer** (has `Float16` since OCaml 5.2 — fine)
> with **the Ctypes layer** (does not — not fine). §3.1's probe only exercised the
> first. This is the generalizable failure: *a type existing in the stdlib says
> nothing about it existing in every library that pattern-matches on that type's
> GADT.* Ctypes' `bigarray_kind` GADT is a closed, hand-written mirror of Bigarray's,
> and mirrors go stale.
>
> **What the implementation had to do instead** — two kind-independent helpers in
> `sarek/core/Memory.ml` (mainline: `bigarray_elem_size` at `:22`;
> `bigarray_void_ptr` added by #290 at `:94`), then patch *every* allocation and
> transfer site:
>
> | Backend | sites patched |
> |---|---|
> | CUDA | `Cuda_api.ml` alloc + H2D + D2H |
> | HIP | `Hip_api.ml` alloc + H2D + D2H (+ keepalives) |
> | OpenCL | `Opencl_api.ml` **two** alloc sites + H2D + D2H; `Opencl_plugin_base.ml` `CL_MEM_USE_HOST_PTR` |
> | Vulkan | `Vulkan_api_memory.ml` alloc + **four** transfer sites — direct-map *and* **staging**, both directions |
> | Metal | `Metal_api.ml` memcpy in/out; `Metal_plugin_base.ml` alloc + both transfers |
> | shared | `Ctypes_ops.ml`, `Vector_transfer.ml`, `Transfer.ml` |
>
> That is materially more surface than "no change needed" — and more than a first
> pass found: the Ctypes workaround had originally landed on HIP only, and Vulkan's
> staging paths and `Metal_api` were the sites a per-backend eyeball missed.
>
> **And it exposed a hole nothing else exercised:** `Execute.ml`'s exec-arg `set`
> (mainline `sarek/execute/Execute.ml:128`; f16 arm at `:158-165`) had no `Float16`
> arm, so **an f16 kernel could not run on the Interpreter *device* at all**. The
> existing f16 tests used `run_interpreter_vectors`, which bypasses that dispatch.
> A new e2e gate (`sarek/tests/e2e/test_f16_host_path.ml`) now runs the round-trip
> on every live device; reverting the OpenCL and Vulkan sites turns all of them into
> `RAW CTYPES FAILURE ... Failure("Unsupported bigarray kind")` — red-on-mutation.
>
> **A second-order trap, worth its own note.** `Ctypes.bigarray_start` returns a
> **managed** fat pointer (`~managed:(Some (Obj.repr ba))`) and ctypes' FFI keeps the
> bigarray rooted across the call. The obvious workaround — take the address and
> wrap it with `Ctypes.ptr_of_raw_address` — is **unmanaged**, and silently drops
> that root. Deterministic probe, major GC with the pointer live: f32 via
> `bigarray_start` → `freed = 0`; f16 via `ptr_of_raw_address` → `freed = 1`; f16
> with the root reconstructed → `freed = 0`. In the D2H direction that is a device
> write into possibly-freed memory. The shipped `bigarray_void_ptr` therefore
> reconstructs ctypes' own fat-pointer shape over `Ctypes_bigarray.unsafe_address`
> rather than using the public unmanaged constructor, and callers that convert back
> to a bare `nativeint` via `raw_address_of_ptr` carry an explicit
> `Sys.opaque_identity` keepalive. A committed test (`test_float16.ml`, `gc_roots`)
> pins both halves of the asymmetry plus a canary that fails the day ctypes grows a
> `Float16` kind — at which point the whole helper collapses back to
> `bigarray_start`.
>
> **The reusable rule:** when a new element type has to cross an FFI boundary,
> enumerate the FFI sites *first*, per backend, including staging/indirect paths —
> do not infer from "the transfer is byte-oriented" that the transfer is free.

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

> **CORRECTION + HELD.** The `Sarek_lower.ml` anchor is dead (file deleted by #291);
> annotation resolution for the live path is `Sarek_types.ml:458`
> (`TEConstr ("float16", []) -> t_float16`). `"half"` was **not** added as an alias.
>
> **HELD, and it is the best call in this spec.** Rather than *add* f16 to the
> numeric lattice, slice 1 **excluded it**: f16 appears in neither `is_numeric`
> (`Sarek_types.ml:416-420`) nor `is_float` (`:430-431`) nor
> `float_literal_can_link` (`:200-201`). "Compute in f32" (§6.2) therefore stops
> being a convention a user can violate and becomes a **type-system guarantee** —
> there is no f16 arithmetic to get wrong, only conversions.
>
> The price is a bounded special-case set — **three enforcement sites**, which an
> architect review judged genuinely bounded rather than the start of a rash:
> 1. `check_numeric`'s explicit `TReg Float16 -> Error [Float16_operand …]` arm, so
>    the diagnostic says "float16" instead of "expected int32"
>    (`sarek/ppx/Sarek_typer.ml:98-103`);
> 2. `reject_float16` for the operator families that skip `check_numeric` — `Eq`/`Ne`
>    and the boolean/bitwise ops (`Sarek_typer.ml:118-124`);
> 3. a deferred guard in `unify`: a tvar that once stood in an operand position is
>    registered "never float16", so it cannot later resolve to f16
>    (`Sarek_types.ml:222-224`).
>
> Sites 2 and 3 were **review findings, not design** — MF4 in review round 1. `Eq`/`Ne`
> skipped every check, so `a.(i) = b.(i)` on a `float16 vector` **compiled** and
> emitted `==` on `__half`; and `check_numeric`'s `TVar _ -> Ok ()` left an
> unresolved operand unchecked forever. **A type-system exclusion is only as good as
> its coverage of the paths that bypass the checker.** Enumerate them.

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
  `Sarek_ir_ptx_expr.ml:530` *(approx. — `:530` is a stale anchor; on mainline it
  lands on an unrelated intrinsic name table. The f16-relevant PTX region is
  `emit_cast`, around `Sarek_ir_ptx_expr.ml:1166-1180`)* +
  `Sarek_ir_ptx_types.ml:175-187`, and float64 analysis
  `Sarek_ir_analysis.ml:181`. Each needs a `TFloat16` arm and f16↔f32↔f64 promotion
  rules (currently a binary "is float64" question becomes 3-way — see §6.2).

> **HELD.** `ECast` was the right carrier, and the "binary becomes 3-way" prediction
> was right enough that the analysis layer was subsequently **parameterized**:
> `Sarek_ir_analysis` now carries a `type feature = Float64 | Float16`
> (`spoc/ir/Sarek_ir_analysis.ml:191`) with one generic detector family and the
> twelve per-width names kept as one-line aliases, plus
> `kernel_requirements : kernel -> feature list` (`:254`). The original shipped as
> two structurally identical 32-line detector families before review collapsed them.
> **Adding bf16 at this layer is now a constructor, one arm, and one line.**

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

> **CORRECTION (implementation diverged; the divergence is an improvement).**
> The "represent f16 as OCaml `float`" half held. The "**add a `VFloat16`**" half
> did **not**: there is deliberately **no `VFloat16` and no `Float16_type`**. An f16
> value is carried as `VFloat32` throughout the interpreter
> (`sarek/interp/Sarek_ir_interp_eval.ml:226`,
> `sarek/interp/Sarek_ir_interp.ml:306`).
>
> The reason is the §3.4 exclusion: because f16 is not in `is_numeric`/`is_float`,
> **there is no f16 arithmetic for the interpreter to dispatch**, so a distinct
> value variant would carry no information — it would only create a second way to
> represent a number and a new set of arms to keep in sync. The one operation f16
> actually needs is "narrow this f32 to binary16", which lives in a single small
> module (`sarek/interp/Sarek_float16.ml`, the deliberately-tiny sibling of
> `Sarek_float32`). Note its documented contract: the narrowing **double-rounds**
> (f64 → f32 → binary16), observable at the overflow boundary —
> `65519.999999999` gives `infinity`.
>
> **Generalizable:** a storage-only type does not need a value-domain representation
> of its own. Adding one is the reflex; resisting it is what keeps the special-case
> set at the three sites of §3.4. The same should hold for bf16.

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
| **CUDA / HIP** | `sarek/codegen/Sarek_ir_cuda.ml:46` | `"float"` / `"double"` (`:49-50`) | `half` (or `__half`) | none — `__float2half(x)` | add `#include <cuda_fp16.h>` to `cuda_header` (`:656` → now `:631`) — **WRONG, see below** |
| **OpenCL** | `Sarek_ir_opencl.ml:57` | `"float"` / `"double"` (`:60-61`) | `half` | suffix `h` (`1.5h`) | prepend `#pragma OPENCL EXTENSION cl_khr_fp16 : enable` — mirror `generate_with_fp64` (`:755-759`) |
| **GLSL / Vulkan** | `Sarek_ir_glsl.ml:226` | `"float"` / `"double"` (`:229-230`) | `float16_t` | suffix `hf` | add `uses_float16` to `glsl_header` (`:1078`) → `#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require` (+ likely `GL_EXT_shader_16bit_storage` for buffer I/O) |
| **WGSL** | `Sarek_ir_wgsl.ml:133` | `"f32"` / **rejects f64** (`:141-143`) | `f16` (native) | suffix `h` | prepend `enable f16;` to **module top** (before bindings, not in `wgsl_header`); do NOT reject in `params_have_float64` (`:1171`) |
| **Metal** | `Sarek_ir_metal.ml:46` | `"float"` / `"float"` (f64→float, `:49-50`) | `half` (native) | suffix `h` | none — native in `metal_stdlib` (already `#include`d `:909`) |
| **PTX** | `Sarek_ir_ptx_types.ml:172` | `.f32` / `.f64` (`:175-176`) | `.f16`, new reg class `%h` | `mov.f16 %r, 0H%04X;` (bit pattern) | none — default `sm_86` ≥ sm_53 |

> ### CORRECTION — the `#include <cuda_fp16.h>` row was wrong twice
>
> **VERIFIED BY EXECUTION** (PR #290; review-round-1 finding MF1). Putting
> `#include <cuda_fp16.h>` unconditionally into `cuda_header` fails in two
> independent ways, and the table's single "add an include" cell hid both.
>
> **(a) It would break every HIP kernel.** HIP reuses the CUDA codegen *verbatim*
> (`Hip_plugin.ml:36`, `Hip_shared.ml:12-13`) — which the spec noted, without
> drawing the conclusion. Probed on gfx1100 through hiprtc: ROCm ships **neither**
> `cuda_fp16.h` **nor** `hip/hip_fp16.h` (both file-not-found), while bare `__half`
> / `__float2half` **do** compile as built-ins. So the include must be emitted under
> a negative HIP guard. Shipped shape (`Sarek_ir_cuda.ml:673-677`):
>
> ```c
> #if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
> #include <cuda_fp16.h>
> #endif
> ```
>
> …selected by `cuda_header_for` (`:679-684`), which prepends it only when
> `Sarek_ir_analysis.kernel_uses_float16 k`.
>
> **(b) It did not work on CUDA either.** **nvrtc is a library, not a driver**: it
> has no default include path, and `__half` is not one of its built-ins. Feeding the
> byte-exact generated source to libnvrtc 13.3:
>
> - as shipped → `NVRTC_ERROR_COMPILATION`, *"could not open source file
>   `cuda_fp16.h` (no directories in search list)"*;
> - with `-I /opt/cuda/targets/x86_64-linux/include` → `NVRTC_SUCCESS`, PTX
>   containing `cvt.rn.f16.f32`.
>
> `Cuda_nvrtc.ml` passed no `-I` at all. It now discovers CUDA include directories
> (`SAREK_CUDA_INCLUDE` override, then `CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT` derived
> `include` and `targets/<triple>/include`, then conventional roots — each required
> to actually contain `cuda_fp16.h`, so a stale `CUDA_PATH` never becomes an `-I`)
> and passes them as `--include-path` on every compile attempt
> (`sarek-cuda/Cuda_nvrtc.ml:308`, `:343-345`, `:387`, `:411`).
>
> ### The generalizable lesson — record this one prominently
>
> **`nvrtc` and `ptxas` are HOST-SIDE compilers.** They need no NVIDIA GPU. "We have
> no NVIDIA hardware on the dev box" was never a reason to leave generated CUDA
> uncompiled, and it is why this shipped: the pre-existing golden test matched
> *substrings of emitted text* and never compiled anything. A codegen backend whose
> only gate is string matching is untested. The new gate
> (`sarek-cuda/test/test_cuda_f16_nvrtc.ml`) compiles the generated source through
> libnvrtc and asserts `cvt.rn.f16.f32` in the resulting PTX, skipping cleanly when
> libnvrtc or the headers are absent. Red on mutation: dropping the include flags
> reproduces the exact `NVRTC_ERROR_COMPILATION` above. **The same argument applies
> to every backend with a host-side compiler available — treat "no device" as a
> reason to skip *execution* tests, never *compilation* tests.**

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

> **HELD — and acted on.** The assessment was correct enough that slice 1 shipped
> with **PTX rejecting f16 outright** rather than half-supporting it, via a shared
> `Sarek_ir_codegen.reject_feature` (`spoc/ir/Sarek_ir_codegen.ml:49`) that every
> backend partially applies. `Sarek_ir_ptx_types` keeps its own raiser so that one
> f16 message (naming the `%h` register-class audit) covers the whole backend. The
> stale anchors: `Sarek_ir_ptx_expr.ml:436/440/530` were re-checked and **do not
> point at f16-relevant code on mainline** — treat them as void; `:1166` and
> `:1177-1180` (`is_f64` prefix sniffing, `emit_cast`) are the real region and still
> resolve. `Sarek_ir_ptx_types.ml:172,175-187` are still exact.

**Per-plugin duplication.** `sarek-cuda/`, `sarek-opencl/`, `sarek-hip/`,
`sarek-metal/`, `sarek-vulkan/` each ship a `Sarek_ir_<backend>.ml` that mostly
`include`s the shared codegen but must be audited for local elttype matches. HIP
reuses CUDA codegen verbatim (`sarek-hip/Hip_plugin.ml:36`, `Hip_shared.ml:12-13`).

> **Anchor note.** PR #77 factorized the C-family codegen into `Sarek_ir_codegen`
> and shifted everything below ~line 600 in the emitters. Current: CUDA
> `cuda_header` `656 → 631`; OpenCL `generate_with_fp64` `755-759 → 713-717`; Metal
> `generate` `909 → 875` (the `#include <metal_stdlib>` is `:882`); WGSL `:1233` is
> **past EOF** (the file is 1227 lines) — treat it as void, `abi_of_kernel` is the
> intended target. GLSL's `:1078`, and the CUDA/OpenCL/Metal/GLSL/WGSL *type-string*
> and *constant-emitter* anchors, are all still exact.

---

## 5. Blast radius — exhaustive matches that adding a constructor forces

Adding `TFloat16` / `EFloat16` / `Float16` / `VFloat16` turns every exhaustive
`elttype` match into a compile error until extended. This is the checklist (the
compiler is the driver — non-exhaustive-match warnings enumerate the rest). ~40
sites, grouped:

> **CORRECTION (measured).** The realised count is **~20** compiler-forced
> exhaustive match sites, not ~40 — and the list below is the wrong *shape* of
> estimate, because it counts match arms and the actual slice-1 diff (87 files,
> ~4,100 insertions) was dominated by things this list does not contain: the FFI
> patching of §3.3, the nvrtc include machinery of §4, three type-system holes, and
> tests. **Match-arm counting systematically underestimates a new element type.**
> Two entries below are now void: `Kirc_Ast.ml` and `Sarek_lower.ml` were deleted by
> PR #291.

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

  > **RESOLVED / HELD.** `float16` only; no `half` alias. `"half"` stays *reserved*
  > (`Sarek_reserved.ml:130`) so the alias remains available without committing to
  > it. The no-literal call held: there is no `CFloat16`, and the analysis layer
  > states that asymmetry explicitly (`const_uses Float16` is false *by
  > construction*, not by a missing case) rather than leaving it implicit.

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

> **HELD — the single best-validated decision in this spec.** Store-as-f16 /
> compute-in-f32 / round-on-store survived implementation untouched, and it is what
> made the two-oracle discipline work: `test_hip_f16` asserts **bit-identical**
> agreement between the interpreter and gfx1100 (13/13). It also went further than
> recommended — see §3.4: because f16 is excluded from `is_numeric`/`is_float`, the
> "compute in f32" rule is enforced by the typer, not by convention. That is the
> shape to reuse for bf16 (§11).

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

> ### CORRECTION — the `%sarek_intrinsic Float16` module is not buildable
>
> **VERIFIED** (PR #290). The *decision* held — explicit conversions, no implicit
> coercion, lowering to `ECast`. The *mechanism* did not. This section proposed
> mirroring `Sarek_float64/Float64.ml:23-24`:
>
> ```ocaml
> let%sarek_intrinsic float64 = {device = (fun _ -> "double"); ctype = Ctypes.double}
> ```
>
> `%sarek_intrinsic` type registration **requires a `ctype`**, and — the same root
> cause as §3.3 — **Ctypes has no half**. There is nothing to put in that field, so
> a `Float16` stdlib intrinsic module cannot be written at all.
>
> **What shipped instead:** the two conversions are **core primitives**, not stdlib
> intrinsics — `float16_of_float32` and `float32_of_float16`
> (`sarek/ppx/Sarek_core_primitives.ml:595`, `:603`; category `"conv_f16"`,
> `t_fun [t_float32] t_float16`) — special-cased in `Sarek_lower_ir.ml:727-732` to
> lower directly to `Ir.ECast (Ir.TFloat16, _)` / `ECast (TFloat32, _)`. Lowering
> to `ECast` rather than to an opaque intrinsic call is also what lets
> `kernel_uses_float16` *see* the conversion.
>
> Note the naming departure from the spec's `f16_of_float` / `float_of_f16`: the
> shipped names are explicit about **both** ends, which matters once bf16 exists.
> **Lesson: check that the extension mechanism you are reusing can actually
> represent the new thing before writing it into the spec — `%sarek_intrinsic` was
> shaped by the types that already existed.**

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

> ### CORRECTION — "⚠ minor" was a HIGH-severity silent-corruption bug
>
> **VERIFIED BY EXECUTION** on gfx1100 (PR #290, review round 2). Reusing
> `get_f32` for vectors held. But framing scalar-by-value f16 params as an optional
> nice-to-have meant nobody **closed the door**, and an unclosed door is not a
> deferral — it is an accepted input.
>
> `lower_param`'s gate rejected `TTuple`, `TFun`, arrays-of-tuples and
> vectors-of-functions, but let `TReg Float16` through. So a scalar `float16`
> parameter **type-checked**; CUDA/HIP then mapped it to a by-value `__half` formal;
> and `Execute.vector_arg` has no float16 constructor, so the only way to supply it
> is `Float32 f` — a 4-byte C float whose **low 2 bytes** the device reads as a
> `__half`. Executed:
>
> ```
> fun (out : float16 vector) (s : float16) -> out.(tid) <- s   with  Float32 3.14159
>   HIP (gfx1100) →  0.000476837158
>   Interpreter   →  3.140625        (correct)
>   errors raised →  none
> ```
>
> The two oracles disagreed silently — precisely the property the f16 test suite
> exists to guarantee. Now a located `TReg Float16` arm in `lower_param` naming the
> parameter and the remedy, with a negative test
> (`sarek/tests/negative/test_f16_scalar_param.ml`).
>
> A sibling MEDIUM from the same round is worth recording because of *where* it
> landed: an f16-vector kernel launched with **float32 vectors** also silently
> misread the buffer (`1 2 3 4` came back `256 512 1.35632e-19 …` on the 7900 XTX,
> no error). No OCaml type constraint can police this — `Execute.vector_arg`'s `Vec`
> constructor is existential and erases the element type before launch. The check
> therefore belongs where the declared type and the supplied vector actually meet:
> a launch-time `Execute.check_vector_element_types`, covering every element type,
> not just f16.
>
> **Two reusable rules.** (1) *A deferred surface must be explicitly rejected, not
> merely unimplemented.* (2) *When a new width is narrower than an existing one,
> every place the wider one can be substituted is a silent-truncation site* —
> enumerate them before shipping.

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

> ### CORRECTION — §7 was not built in slice 1, and only half of it should be
>
> **Status after PR #290:**
>
> | §7 proposal | Shipped? |
> |---|---|
> | `kernel_uses_float16` detector | **Yes** — and generalized past the spec: `type feature = Float64 \| Float16` with one parameterized detector family (`Sarek_ir_analysis.ml:191`, `:250`, `:254`) |
> | pragma/extension/`enable` prepends driven by it | CUDA only (`cuda_header_for`); the rest are slice 2 |
> | `supports_fp16` in `Framework_sig.capabilities` (`:26-41`, `supports_fp64` at `:34`) | **No** |
> | `Device.allows_fp16` / `with_fp16` (mirroring `Device.ml:129-132,186`) | **No** |
> | launch-time gate, hard error | **No** |
>
> **Why the deferral is the right call, not a gap.** Slice 1 supports f16 on exactly
> two backends (CUDA/HIP); the other five *reject f16 at codegen* through the shared
> `Sarek_ir_codegen.reject_feature`. A **codegen-time** rejection is strictly better
> than a launch-time capability gate wherever the answer is statically known: it
> fires earlier, needs no device, and cannot be reached with a wrong device
> selected. `supports_fp16` earns its place only once a backend supports f16 *for
> some devices and not others* — i.e. once slice 2 lands OpenCL (`cl_khr_fp16` is
> genuinely optional) and Vulkan (`shaderFloat16`). Until then it is a field nothing
> can read a false value from.
>
> **The hard-error-vs-fallback question is therefore still open**, but its scope
> shrank: the recommendation of hard error stands, and #64 should decide it for the
> whole `feature` set at once rather than per width — which is what
> `kernel_requirements : kernel -> feature list` (`Sarek_ir_analysis.ml:254`) was
> added to enable.
>
> **Generalizable:** *prefer a static rejection at the layer that knows, over a
> dynamic gate at the layer that runs.* The spec reached for the f64 machinery
> because it was the visible template, without asking whether f64's problem
> (universally-supported-but-optional) is f16's problem (not yet implemented
> everywhere). It is not.

---

## 8. Slice plan

f16 **scalar** is the prerequisite; f16 **matrix/fragment** types are the follow-on.
Slices are independently reviewable and each ends at a green build + tests.

> **HELD — the slice plan is the part of this spec that needed no revision.** The
> boundaries drawn here (1 = host + type system + native/interp + CUDA/HIP; 2 =
> remaining backends with PTX carved out as its own unit; 3 = matrix/fragment types
> as the actual tensor-core unlock) were all load-bearing during implementation, and
> carving PTX out of slice 1 is what let the review focus on the FFI and
> type-system problems instead. **Slice 1 shipped as PR #290.** The one adjustment:
> the capability work listed under slice 1 moved to slice 2, for the reason in §7.

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

> ### CORRECTION — "each is mechanical except WGSL" is refuted for OpenCL
>
> **VERIFIED BY EXECUTION** (#57 slice 2a, 2026-07-26). The OpenCL *codegen* is
> indeed mechanical — `"half"`, a narrowing arm, and a `cl_khr_fp16` pragma.
> That was never the binding constraint, and pricing this slice by codegen
> surface repeated §5's error one width later: **the cost of a backend is not
> the cost of its type string.**
>
> On rusticl/radeonsi the ACO backend fuses the f32 multiply into the f32→f16
> narrowing that consumes it — rounding **once** where §6.2's discipline
> mandates twice. Exhaustive sweep, all 63488 finite binary16 inputs, on **two**
> devices (RX 7900 XTX / navi31 and the Raphael iGPU / gfx1036): **620/63488**
> disagreements, first at `x=5.68359375`. That is the *same defect and the same
> count* as HIP/AMDGPU — unsurprising in hindsight, since both are ACO, which
> nobody predicted because the slice plan grouped backends by *language* rather
> than by *backend compiler*.
>
> The difference from HIP is that HIP's fix does not transfer. Measured
> non-fixes, all still 620/63488: `#pragma OPENCL FP_CONTRACT OFF`, a `volatile`
> local, a `volatile __private` pointer, an `as_half`/`as_ushort` bitcast
> round-trip, `convert_half_rte`. HIP's `asm volatile("" : "+v"(x))` does not
> even compile — rusticl goes through SPIR-V, where AMDGPU register constraints
> do not exist. Only a `volatile __global` and a `volatile __local` round-trip
> work (both **0/63488**, which is also the liveness control proving the sweep
> can go green), and both cost memory traffic per narrowing.
>
> **Outcome: OpenCL f16 stays rejected**, now with a measured justification
> instead of a "not yet implemented" placeholder. Shipping it would have meant a
> backend silently disagreeing with the interpreter on 620 inputs — exactly the
> §6.4 failure mode, reintroduced at a different layer.
>
> **Two generalizable rules.**
> 1. *Group backends by their backend compiler, not by their source language.*
>    OpenCL and HIP look maximally different at the source level and are the same
>    compiler underneath. GLSL/Vulkan on RADV is a third front end onto ACO and
>    should be assumed to carry this defect until measured.
> 2. *Port the property, not the patch.* The HIP work delivered a barrier; what
>    was actually reusable was the exhaustive-sweep harness and the bit-identity
>    property. The barrier itself did not survive a change of front end.
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

> ### CORRECTION — the cost model was wrong in both directions
>
> This is the most useful correction in the document for whoever adds the next
> width. **The duplication tax is real, but it is not where this spec put it.**
>
> **Direction 1 — `Kirc_Ast` was priced far too high (measured).** `Kirc_Ast.elttype`
> had **4 constructors and 4 use sites** across the entire tree, exactly **one** of
> which a new width forces (`lower_reg_elttype`). One line. Slice 1 spent that line
> on a *rejection*, since the path was already dead. **PR #291 (`ba375a36`, #78) has
> since deleted `Kirc_Ast.ml` and `Sarek_lower.ml` entirely (~2,215 lines).** It is
> no longer a consideration. The "roughly halve slice 1–2 cost" estimate above was
> off by an order of magnitude, and a later follow-up investigation repeated the
> same overestimate — worth noting, because the error survived a second look. The
> lesson is mechanical: **count the constructors and grep the use sites before
> pricing a vocabulary.** Nobody did, on either pass.
>
> **Direction 2 — the real tax was omitted entirely.** It is the
> `spoc/ir/Sarek_ir_types.ml` (runtime) ⇄ `sarek/ppx/Sarek_ir_ppx.ml`
> (compile-time) **mirror**, plus the `sarek/transpile/Sarek_ir_conv.ml` bridge that
> exists only to connect them. Every new width is defined twice and converted once.
>
> **And an architect review established that the mirror has no dependency
> justification.** The stated rationale is "compile-time types vs runtime types",
> which would hold only if the PPX could not link the runtime IR. It can, and does:
>
> - `sarek/ppx/dune` declares `(library (name sarek_frontend) … (libraries ppxlib
>   sarek_ir))` — **the frontend that defines the mirror already links
>   `sarek_ir`.**
> - `spoc/ir/dune` declares no `(libraries …)` at all: `sarek_ir` is a
>   dependency-free pure library. There is no cycle to break.
>
> Normalising and diffing the two files: `memspace`, `binop`, `unop` and `const` are
> **character-identical**; `elttype` differs only in doc comments. The two
> representations diverge at **exactly one node** — `SNative`, whose payload is
> `Ppxlib.expression` in the mirror (`Sarek_ir_ppx.ml:127-130`) and closures at
> runtime (`Sarek_ir_types.ml:160-163`). One node's payload does not justify a
> duplicated type universe.
>
> **The cheap first step**, therefore, is to share the **leaf vocabularies** by type
> equation rather than by copy:
>
> ```ocaml
> type elttype = Sarek_ir_types.elttype = TInt32 | TInt64 | ...
> ```
>
> …and the same for `memspace`, `const`, `binop`, `unop`. This makes the
> corresponding `conv_*` bridge functions (`Sarek_ir_conv.ml:21`, `:26`, `:52`,
> `:60`, `:80`) the identity, removes roughly **4–5 of the ~20** exhaustive match
> sites a new width forces, and **adds no dependency** — the link already exists.
> `SNative` and the expression/statement layers stay mirrored.
>
> **Measured cost of adding f16, for calibration:**
>
> | | |
> |---|---|
> | compiler-forced exhaustive match sites | **~20** (spec estimated ~40) |
> | review round 1 — must-fix findings | **4**: unresolvable nvrtc include; host FFI path unpatched on four backends; a GC-root regression (a reconstructed *unmanaged* ctypes pointer stripped the root `bigarray_start` normally provides); two type-system holes (`Eq`/`Ne` bypass, unresolved-tvar bypass) |
> | review round 2 | duplication collapse (analysis detectors, backend rejections) + **1 HIGH** (scalar f16 param silently truncating — §6.4) + 1 MEDIUM (f32 vector supplied to an f16 param) + 2 LOW |
> | diff | 87 files, ~4,100 insertions |
>
> Note the distribution: **none of the four round-1 must-fixes and neither of the
> two soundness bugs were in a match arm.** They were in the FFI layer, the host
> compiler invocation, the GC contract, and the parameter gate. A blast-radius
> estimate built from `grep`ing exhaustive matches will predict the *easy* part of a
> new element type with reasonable accuracy and the *hard* part not at all.
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

*Answers below are from the slice-1 implementation and its reviews.*

1. **Legacy IR:** is `Kirc_Ast` still required, or may f16 target only `Sarek_ir`?
   (Halves the blast radius if the latter.)
   > **ANSWERED: not required, and gone.** It was fully dead. f16 rejects it; PR
   > #291 deleted it. It was also never worth half the blast radius — one line (§9).
2. **Strategic:** is slice 3 (tensor cores) funded? If not, is f16 storage/interchange
   alone worth slices 1–2? (§9.)
   > **ANSWERED: the coopmat-first tensor-core strategy is the funded direction**,
   > which is what makes bf16 rather than f8/f4 the next width (§11).
3. **Launch gate policy:** hard error vs warn-and-emulate on an f16-lacking device
   (§7). Composes with #64.
   > **STILL OPEN, scope reduced.** Slice 1 needs no launch gate — unsupported
   > backends reject at codegen. Decide it in #64 for the whole `feature` set at
   > once (§7).
4. **Surface taste:** `half` alias alongside `float16`, or `float16` only? (§6.1.)
   > **ANSWERED: `float16` only.** `half` stays reserved but unbound.
5. **f16 scalar params:** are by-value f16 kernel params in scope for slice 1
   (→ `NA_Float16`), or vectors-only first? (§6.4.)
   > **ANSWERED: vectors only — and the scalar form is now an explicit compile
   > error.** Leaving it merely unimplemented was a HIGH-severity silent-corruption
   > bug (§6.4).

---

## 11. What comes after f16 (added post-implementation)

Decisions taken since this spec was written. **Reasoned, not yet implemented** —
these are directional, not verified.

### 11.1 bf16 is the next width. Not FP8, not FP4.

- **bf16 needs no new machinery.** It has f32's **8-bit exponent**, so it shares
  f32's dynamic range: there is **no scaling infrastructure to build**. Under the
  store-as-bf16 / compute-in-f32 rule (§6.2), a bf16→f32 conversion is an exponent-
  preserving widen, and the type-system exclusion of §3.4 applies verbatim. bf16 is
  as close to "f16 again, with a different `cvt`" as a new width can be — which is
  exactly why it is the right second instance for learning what generalizes.
- **It fits the funded strategy.** bf16 is among the supported component types of
  `VK_KHR_cooperative_matrix`, so it lands inside the coopmat-first tensor-core path
  (#62) rather than beside it.
- **Hardware coverage is broad:** NVIDIA Ampere and later, AMD RDNA3 / CDNA2 and
  later, Apple, Intel.
- **Local testability caveat, stated up front.** RADV on the RX 7900 XTX advertises
  `VK_KHR_shader_float16_int8` but **no bfloat16 extension**, even though RDNA3 has
  bf16 in its ISA. So the dev box can cover bf16 on the HIP/CUDA path but not
  necessarily through Vulkan — plan the bf16 gates accordingly, and remember §4's
  lesson: host-side compilation gates (hiprtc, nvrtc, ptxas, glslang) are available
  regardless of what the device advertises.

### 11.2 FP8 and FP4 are deliberately deferred — and are not element types

This is a scoping decision, not a backlog item. Both are the wrong *shape* for the
`elttype` vocabulary:

- **FP4 is a block format by definition.** MXFP4 is 32 values plus a shared E8M0
  scale; NVFP4 is 16 values plus a shared FP8 scale. **The scale is part of the
  layout.** A scalar FP4 is not a meaningful thing to add to `elttype` — the unit of
  storage is the block, so the DSL concept required is a block-scaled tensor type,
  which is a different design problem from a scalar width.
- **FP8 without scaling hands users silent garbage.** Usable FP8 training needs
  per-tensor scaling (E4M3 forward, E5M2 for gradients, amax history for the scale
  update). That machinery is why NVIDIA ships Transformer Engine as a separate
  library rather than shipping a type. Adding an unscaled FP8 element type would
  reproduce the §6.4 failure mode at scale: it would type-check and produce
  plausible-looking wrong numbers.
- **Their throughput lives in a different programming model.** FP8/FP4 speed comes
  from `wgmma` / `tcgen05` with swizzled shared-memory layouts, descriptor operands
  and asynchronous copies (TMA). That is not an added type; it is a different way of
  writing a kernel. **`VK_KHR_cooperative_matrix` does not cover them**, so they are
  also outside the cross-vendor path Sarek's value proposition depends on.

Revisit FP8 only alongside a scaling design, and FP4 only alongside a block-format
design.

### 11.3 Do NOT generalize width-addition yet

The tempting move after slice 1 is a table-driven `float_width` descriptor —
one record per width (bit count, Bigarray kind, per-backend type string, per-backend
`cvt`, feature name) driving every site generically. **The architect's advice, taken:
don't, not from one instance.**

- Do the **shared-leaf** step of §9 and the duplication fixes already identified
  (the analysis-layer `feature` parameterization and the shared
  `Sarek_ir_codegen.reject_feature` are both done and are the model).
- **Add bf16.** Let the *second* width reveal which rows are genuinely tabular and
  which only looked tabular from one example. Two instances is the minimum for
  distinguishing a pattern from a coincidence, and bf16 is deliberately the *most*
  similar next width — if a row does not generalize across f16 and bf16, it will not
  generalize at all.
- **Be honest about the ceiling.** Such a table would cut maybe **60%** of
  per-width cost, not 100%. The residue is type-level and a value-level table cannot
  express it:
  - the GADT arm `Float16 : (float, Bigarray.float16_elt) scalar_kind` — the
    `Bigarray.*_elt` phantom is a distinct *type*, not a value;
  - the `Bigarray.Float16` matches in `Execute.ml` and the plugin bases, which are
    matches on a GADT constructor and refine types in each branch.
  Budget the next width as "meaningfully cheaper", not "free".

---

*Anchors were verified against the worktree at original commit-time and re-verified
during the 2026-07-25 amendment; drifted anchors carry their current line, and
anchors that could not be re-verified are marked `(approx.)` or called out as void.
The `float16_elt` / `Bigarray.Float16` availability claim in §0/§3.1 was verified by
compiling and running a probe in this OCaml switch, not assumed — but see the §3.3
CORRECTION for what that probe did not cover.*
