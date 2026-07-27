# PTX aggregate layout (records / variants / tuples)

Operational specification of the byte layout used by the PTX direct emitter for
aggregate values, introduced by the `ptx-records-variants` task (2026-07-22).
Single source of truth in code: `spoc/ir/Sarek_ir_layout.ml` (all offsets, sizes,
strides — no emission site computes its own offsets). Formal mirror:
`formal/codegen-ptx/theories/PtxLayout.v` (12 theorems, 0 admits) + conformance suite
`formal/codegen-ptx/test/test_layout_conformance.ml` (32 063 shapes, zero divergence).

**L8 update (2026-07-23):** the host aggregate ABI moved from PACKED to ALIGNED
(C-ABI-compatible). This unlocked mixed-alignment records (`{i32;f64}`) and
f64/i64-payload variants on all backends, and resolved the host-vs-C-compiler
landmine below. Zero-breakage: every previously-shipped `[@@sarek.type]` is
homogeneous 4-byte, and aligned == packed byte-for-byte for those.

## Layout rules (aligned, C-ABI-compatible)

- **Records:** each field is placed at the next offset satisfying its natural alignment
  (padding inserted); total size is rounded up to the struct's max member alignment.
  Mirrors the host PPX `aligned_record_offsets` (`sarek/ppx/Sarek_ppx.ml`). Element
  stride of a `t vector` = total aligned size (e.g. `point3d` {x;y;z}:f32 = 12 bytes,
  unchanged; `{i32;f64}` = 16 bytes with a@0, b@8).
- **Variants:** `[tag : int32 at offset 0][payload region at offset P]` where
  `P = max(4, max payload-member alignment)`; element size =
  `round_up(P + max payload size, max_align)`; per-constructor payload arg offsets are
  aligned within the payload region.
  **Tag = constructor declaration index** (same in host PPX `List.mapi`, C-family enum,
  PTX emitter, and proved as `ctor_tag_is_index` in PtxLayout.v).
- **Scalar sizes/alignments** mirror the host mapping exactly: 4 for int32/float32/bool/
  int/unit (`bool = 4` host catch-all), 8 for int64/float64.
- **Tuples** are anonymous records with positional fields `_0/_1/…` (registers only;
  tuples cannot be stored in vectors — no host tuple vector type exists).

## Alignment is now by construction

The aligned layout places every scalar leaf on its natural boundary by construction, so
the old packed misalignment rejection is DEAD: `Sarek_ir_layout.Misaligned_field` is
retained only as a defensive internal invariant guard and can no longer fire for
well-formed input. PtxLayout.v proves this unconditionally (`record_leaf_aligned`,
`variant_leaf_aligned`, `record_always_accepted`, `variant_always_accepted`). Nested
variants below top level and array/vector fields are still rejected (typed
`Layout_error`). Since offsets are 8-aligned when needed, `ld.global.f64` /
`st.global.f64` are emitted at natural boundaries (no UB) — confirmed by
`test_ptx_snapshot.ml` (aligned f64 field) and a ZLUDA `{i32;f64}` round-trip
(`sarek/tests/e2e/test_ktype_mixed_align.ml`, RX 7900 XTX).

## Performance (aligned vs packed)

Aligned layout costs padding bandwidth ONLY for mixed-alignment types — `{i32;f64}` is
16B aligned vs 12B packed (+33%, 4 bytes of padding). Homogeneous-4-byte types (the whole
shipped fleet) are unchanged (same offsets, same size). **Guidance:** order struct fields
largest-alignment-first (f64/i64 before i32/f32) to minimise inserted padding — standard
C struct-packing advice, now directly actionable since the ABI is alignment-aware.

## In-register representation (SROA)

Local aggregate values never touch memory: a record is one register per scalar leaf
(nested records flatten), a variant is a u32 tag register + per-(ctor,arg) registers
(absent ctors' slots = allocated, never-written registers — uniform shape for leaf-wise
merge). `match` lowers to a tag-compare branch chain (never `selp`); payload bindings are
arm-scoped. Vector elements are accessed field-wise:
`mul.wide.u32 idx×stride + base`, then typed `ld/st.global` at immediate field offsets —
a shape pinned on hardware by `sarek-cuda/test/test_ptx_stride_spike.ml` (ZLUDA, RX 7900 XTX).
Whole-element copies emit ALL loads before ANY store (aliasing safety).

## RESOLVED landmine (fixed by L8, 2026-07-23)

Previously the host PACKED layout and the C compiler's natural-alignment layout diverged
for any mixed-alignment aggregate: the CUDA/C, OpenCL and Metal backends emit real C
`typedef struct`s and let the C compiler pad (e.g. `{int32; float64}` → payload at 8,
size 16), while the host custom-vector get/set read/wrote packed offsets (payload at 4,
size 12) — silent data corruption for any such type crossing the host/device boundary on
a C-family backend. **L8 fixed this by migrating the host ABI to aligned**: the host PPX
get/set, `Sarek_ir_layout`, and the C-compiler-aligned struct now agree byte-for-byte
(no C-family backend change was needed — the C compiler already aligned; only the host
and PTX sides caught up). The PTX backend no longer rejects these shapes — it lays them
out aligned and emits natural loads/stores. Verified end-to-end by a ZLUDA `{i32;f64}`
round-trip on RX 7900 XTX.

## Related

- `kb/backends/cuda.md` — C-family variant codegen (shared `gen_variant_def`), nullary-tag
  quirk (separate, still open).
- Interpreter backend uses hash-derived variant tags (collision issue, separate concern) —
  NOT the ABI anchor; the PTX/host tag contract is declaration-index.

## PTX backend limits (quick reference)

Full reference with quoted error messages, file:line citations, and execution-model
rationale: `roster/ptx-limits-campaign/L10-inherent-limits.md`.

| Construct | Status | Workaround |
|---|---|---|
| `EApp` with non-variable callee (function values / dynamic dispatch) | permanent | Restructure so the callee is a statically named top-level helper (`EVar`). |
| Unbounded recursion without `sarek.inline N` | permanent (contract) | Rewrite as tail recursion (auto-transformed to a loop), or annotate with `pragma ["sarek.inline N"]` for a sufficient bounded depth. |
| Tuples stored in global vectors | permanent | Bind with `let` and use components individually, or use a registered record type (`[@@sarek.type]`) instead. |
| Variants nested below top level in aggregates | permanent | Hoist the variant to its own vector, or flatten its payload into the enclosing record as explicit scalar fields + tag. |
| Mixed-alignment aggregates | done — L8 (aligned host ABI) | Supported natively (C-ABI aligned). Optional: order fields largest-alignment-first to minimise padding. See `roster/ptx-limits-campaign/L8-aligned-host-abi.md`. |
| f64 transcendentals (sin/cos/tan/exp/log/log10/sinh/cosh/tanh/pow) | done | None needed — landed via `Sarek_ir_ptx_softmath`. |
| asin/acos/atan/atan2/expm1/log1p (f32 and f64) | done — L2 (in-PR) | None needed — f64 via `Sarek_ir_ptx_softmath` (fdlibm algorithms); f32 computes in f64 and rounds back (`cvt.rn.f32.f64`). |
| Dynamic/non-literal shared memory size | scheduled — L6 | Declare `let%shared` arrays with a compile-time positive integer literal size. |
| Local (per-thread private) arrays | scheduled — L7 | No supported workaround today; avoid indexable local arrays (use named scalars, or move data to global/shared memory with an explicit size). |
| Aggregate payload args in variant vector elements | permanent | Flatten nested-record payload arguments into scalar constructor fields. |
| Shared-memory aggregate (record/variant) arrays | permanent | Use a global vector parameter for record/variant arrays, or restrict shared arrays to native scalar element types. |
| Record/variant kernel params by value | NO-GO — L9 (EC-11 workaround shipped) | Pass fields as separate scalar params, or wrap in a 1-element vector (`Vector.create_custom`). See `roster/ptx-limits-campaign/L9-byvalue-aggregate-params.md`. |
