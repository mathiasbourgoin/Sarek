# Spec — ptx-records-variants

> **Revision history (backlog-205, 2026-07-31).** Validated 2026-07-22 against
> a *packed* layout: record offsets = cumulative `field_byte_size` sums;
> variant payload fixed at offset 4 with element size `4 + max_payload_bytes`;
> any aggregate placing a scalar leaf at a non-naturally-aligned offset was
> rejected. Campaign item **L8** (2026-07-30) migrated the host PPX,
> `Sarek_ir_layout` and `formal/codegen-ptx/theories/PtxLayout.v` from PACKED to
> **ALIGNED (C-ABI)**: each field/payload member is placed at the next offset
> satisfying its own natural alignment (padding inserted), and the total size
> is rounded up to the aggregate's maximum member alignment — the standard C
> struct ABI, byte-for-byte identical to the `typedef struct {...}` the
> C-family backends emit. `PtxLayout.v` now proves `record_always_accepted` /
> `variant_always_accepted`: alignment holds unconditionally, so the old
> rejection case cannot fire for well-formed input (`Misaligned_field` is kept
> only as a defensive internal invariant — see `Sarek_ir_layout.mli`). What is
> still rejected is unchanged in kind: variants nested below top level, and
> array/vector-typed fields or payload members.
>
> This revision brings the normative text into line with that migration:
> **FR-002, FR-003, FR-004, C-2, EC-2**, the US-1/US-2 scope notes, and
> **AC-5/CHECK-5** below are rewritten to state the aligned rule (each carries
> an inline "(was: ...)" note recording the retracted packed wording so the
> history is not lost). The authority for the current rule is the retraction
> block in `formal/codegen-ptx/STATUS.md` ("Retraction — this model is no
> longer packed") together with `PtxLayout.v` and
> `spoc/ir/Sarek_ir_layout.mli`. **Not re-run in this pass:** the ZLUDA e2e
> sweep (CHECK-1/CHECK-2/CHECK-7) and the Rocq/conformance build (CHECK-6) —
> this is a documentation correction, not a fresh end-to-end validation cycle;
> the PTX-generation-level checks (CHECK-3/4/5/8, no GPU required) were re-run
> and pass against the rewritten text.

**Status: REVISED (aligned C-ABI layout, backlog-205; originally VALIDATED 2026-07-22 against the now-retracted packed layout — see Revision history above)**
**Date:** 2026-07-22T19:20:00Z
**Superseded:** 2026-07-30T00:00:00Z
**Revised:** 2026-07-31T00:00:00Z
**Source brief:** `ptx-records-variants-intake`, in the untracked workstation-local `briefs/` tree (VALIDATED, Type: feature, Trust boundary: no)

## Summary

Implement aggregate (record / variant / tuple) support in the Sarek PTX direct emitter.
Aggregates are lowered through a single pure **layout function** (aligned, host-C-ABI-compatible
byte offsets — was: packed byte offsets) with two representations: **SROA register sets** for
local values and **field-wise global memory access** for elements of `t vector` parameters.
Variants use the host tag encoding (`[tag:int32@0][payload@P]`, `P = max(4, max payload member
alignment)`, tag = constructor declaration index). The layout function is mirrored in Rocq
(separate module, existing theorems untouched) with proved lemmas and a hand-mirror CMBT
conformance test.

## Entities

- **`Sarek_ir_layout`** (pure OCaml module): `field_offsets : elttype -> (string * int) list`,
  `sizeof : elttype -> int`, `alignof_scalar : elttype -> int`, plus variant payload offsets.
  Aligned (C-ABI) semantics identical to the host PPX (`sarek/ppx/Sarek_ppx.ml`
  `aligned_record_offsets`:297-311 / `calc_type_size_early`:313-315 for records;
  `variant_payload_offset`:348-350 / `calc_variant_size`:353-358 for variants — was: `calc_offsets`
  packed cumulative sums, variant payload fixed at offset 4). **Does not redefine**
  `ir_shared_decl`/`ptx_kern_shared`/`emit_kernel_correct` from `specs/ptx-dshared-formal.md`.
- **Aggregate SROA value** (emitter-internal): a record value = one PTX register per scalar
  leaf field (nested records flattened recursively); a variant value = one u32 tag register +
  one register per (constructor, payload-arg). Env extended to bind names to scalar registers
  **or** aggregate register sets.
- **`PtxLayout.v`** (new Rocq theory in `formal/codegen-ptx/theories/`): standalone inductive
  mirroring the layout domain; NOT the shared `elttype` of `PtxTypes.v` — the three existing
  theorems compile unmodified.

## User Stories

### US-1: Records on CUDA/PTX (P0)
As a Sarek user, I can run kernels that construct records, read fields, copy whole records
between global vector elements, and read/write record elements of `t vector` params on the
CUDA/PTX backend, with results equal (within each test's existing epsilon) to the native
backend.
**Scope excludes:** record kernel params by value (host marshalling), tuples in global memory.
(Mixed-alignment records are laid out correctly, not rejected — was: rejected, US-5; see the
Revision history banner and US-5 below for what is still rejected.)
**Independent test:** `test_ktype_record` (GPU gate removed), `test_ktype_helper`,
`test_registered_type`, `test_transpose` (point3d), `test_nbody_ppx`, `test_ray_ppx`,
`test_complex_types` — all pass on the CUDA/PTX device.
**Acceptance scenarios:**
1. **Given** a `point vector` src on the XTX via ZLUDA, **When** `dst.(i) <- {x = src.(i).x +. 1.; y = src.(i).y}` runs on CUDA/PTX, **Then** per-device Status is OK and values match native within 1e-3.
2. **Given** a kernel-local record built by an inlined helper (`make_point`), **When** the kernel computes `sqrt(p.x² + p.y²)`, **Then** results match native (helper record args/returns thread through inlining).
3. **Given** a 12-byte `point3d vector` (non-power-of-2 stride), **When** `output.(o) <- input.(i)` whole-record copy runs, **Then** addressing uses general byte-stride multiplication and all three fields survive the copy.

### US-2: Variants + match on CUDA/PTX (P0)
As a Sarek user, I can construct variants (incl. nullary) and `match` on them in expression and
statement position with payload binding on CUDA/PTX; the byte layout is host-compatible.
**Scope excludes:** guards/nested patterns (not in the IR). (Variants with an 8-byte payload
scalar are laid out with the payload region at offset 8, not rejected — was: rejected, payload
offset 4 misaligned; US-5.)
**Independent test:** `test_registered_variant` passes on CUDA/PTX; unit snapshots cover the
general shapes the e2e test doesn't (multi-arg payload, ≥3 constructors, nullary-only).
**Acceptance scenarios:**
1. **Given** `color = Red | Value of float32` in a global vector, **When** the kernel matches and computes `Red -> 0. | Value v -> v +. 1.`, **Then** CUDA/PTX matches native.
2. **Given** an `EMatch` in value position, **When** emitted, **Then** lowering is branch-based (never selp), all arms write the same typed result register (set).
3. **Given** an `SMatch` case `PConstr("Value", [v])`, **When** emitted, **Then** `v` binds payload registers scoped to that arm only (no cross-arm register merge needed).

### US-3: Tuples as anonymous aggregates (P1)
Literal tuples and multi-arg variant constructor payloads lower through the same layout/SROA
machinery (positional slots `_0/_1/…`; byte offsets = aligned cumulative, i.e. each slot at the
next offset satisfying its own natural alignment — was: packed cumulative — matching the host's
multi-field payload layout).
**Scope excludes:** tuples stored in global vectors (no host tuple vector type — rejected with
a clear error).
**Independent test:** unit snapshot tests (tuple construction/projection; 2-arg variant payload
construct + match binding both args).
**Acceptance scenarios:**
1. **Given** `let (a, b) = …` style IR (`ETuple` + `PConstr("tuple", …)`), **When** emitted, **Then** each component lives in its own register and projections are movs.
2. **Given** `Pair of float32 * int32` payload, **When** constructed and matched, **Then** payload slots at offsets 4 and 8 round-trip through a global vector correctly.

### US-4: Layout function frozen in Rocq (P1)
The byte-layout function is one pure OCaml module mirrored by an independent Rocq
re-implementation (option (b) of C-16) with proved lemmas and a hand-mirror CMBT conformance
test (the project's established pattern — see FR-042), so the ABI is frozen formally without
touching the existing three theorems.
**Scope excludes:** re-proving `emit_expr/stmt/kernel_correct` over aggregates (deferred,
documented follow-up).
**Independent test:** `dune build @formal` (or project-local build) proves 0-admit lemmas;
conformance test compares the Rocq hand-mirror and OCaml functions.
**Acceptance scenarios:**
1. **Given** the accepted-layout predicate, **When** lemmas are checked, **Then** field
   non-overlap, in-bounds (offset+size ≤ sizeof), size correctness, AND natural-alignment of
   every scalar leaf (the C-15 side-condition) are proved with 0 admits.
2. **Given** an enumeration of small shapes (≤4 fields over {i32,f32,bool,i64,f64} + variants
   ≤3 ctors ≤2 args) plus qcheck random shapes, **When** the conformance test runs, **Then**
   OCaml `Sarek_ir_layout` and the Rocq hand-mirror agree on offsets/sizes/accept-reject.
3. **Given** `point` and `color` from the e2e tests, **When** compared against the host
   layout constants (literal offset/size pins mirroring `custom_type` elem_size), **Then**
   layouts agree byte-for-byte (live get/set round-trips are exercised by the e2e suite).

### US-5: Precise rejection of out-of-scope constructs (P2)
Out-of-scope constructs fail at codegen with errors naming the construct AND the workaround.
**Acceptance scenarios:**
1. **Given** a bare record kernel param (`DParam` with `TRecord`, no arr_info), **When** codegen runs, **Then** the error names the param and suggests "pass fields as separate scalar params or use a 1-element `t vector`". A `TVec (TRecord …)` param with arr_info is NOT rejected (EC-11 discrimination).
2. **Given** a record `{a: int32; b: float64}` (`a` at 0, `b` at the 8-byte-aligned offset 8,
   stride 16), **When** codegen runs, **Then** it COMPILES: `b` is laid out at its naturally
   aligned offset, not rejected. (Was: rejected as misaligned — `test_mixed_align_record_param_accepted`,
   `sarek/tests/unit/test_ptx_snapshot.ml:2072`.)
3. **Given** a variant with an f64 payload, **When** codegen runs, **Then** it COMPILES: the
   payload region sits at the 8-byte-aligned offset, not rejected. (Was: rejected —
   `test_f64_variant_param_accepted`, `sarek/tests/unit/test_ptx_snapshot.ml:2108`.)
4. **Given** a variant nested below top level (record field or variant payload), or an
   array/vector-typed field or payload member, **When** the layout function runs, **Then** it is
   rejected with a precise error (`Nested_variant` / `Unsupported_field` —
   `sarek/tests/unit/test_ir_layout.ml:229` `test_reject_variant_in_record`,
   `sarek/tests/unit/test_ir_layout.ml:248` `test_reject_vec_in_record`). This is unchanged by
   the packed-to-aligned migration.

## Challenge Resolutions

| C | Resolution |
|---|---|
| C-1 | Aggregate strides use general multiplication (`mul.wide.u32` for global 64-bit addressing; `mul.lo.u32` shared). Scalar pow-2 `shl` fast path unchanged. |
| C-2, C-18, C-20 | **Lay out** every aggregate on the aligned (C-ABI) rule: each field/payload member at the next offset satisfying its own natural alignment, total size rounded to the aggregate's max member alignment. This keeps ABI-match AND alignment soundness *unconditionally* — `PtxLayout.v` proves `record_always_accepted`/`variant_always_accepted`, so no aggregate is rejected for misalignment; `Misaligned_field` is a defensive internal invariant only (`Sarek_ir_layout.mli`). Still **rejected** with a precise error (US-5): variants nested below top level, and array/vector-typed fields or payload members. (Was: reject any aggregate whose *packed* layout placed a scalar leaf at a non-naturally-aligned offset, incl. every 8-byte variant payload at offset 4 — flagged as future work "if the host layout gains alignment"; that revision (campaign item L8) has since landed.) |
| C-3 | Nested records IN scope (layout + SROA are recursive; host `gen_field_read` supports nested customs). Variants nested inside aggregates → rejected with a clear error (untested). |
| C-4 | The blanket `<> "Native"` gate in test_ktype_record is removed; if a specific non-CUDA framework fails, it gets a named, commented gate (decided at implement; AC only requires CUDA/PTX OK). |
| C-5 | Aggregate args/returns through EApp inlining ARE in scope and test-covered (ktype_helper/nbody/ray helpers). `callee_env` binds SROA sets; `inline_ret` supports aggregate result register sets. |
| C-6, C-19 | Register-only SROA justified: PTX registers are virtual; ptxas/ZLUDA performs real allocation and spills to local itself (same mechanism NVCC output relies on). No emitter-side spill path. |
| C-7 | transpose/jit_only already attempts CUDA; failures were masked by report-only exit codes. ACs assert per-device Status OK, not exit codes. |
| C-8 | Unit snapshot tests cover multi-arg payloads, ≥3 ctors, nullary-only variants (US-2/US-3 scenarios). |
| C-9, EC-5 | Emitter emits the last arm as unconditional default when it is `PWild` OR when all constructors are covered (tag ∈ [0,n) guaranteed). Non-exhaustive without wildcard → codegen error "non-exhaustive match". |
| C-10, EC-4 | EMatch is always branch-based (it is already in `expr_needs_branch_guard`); all arms have one post-typing result type, writing one typed result register (set). selp never used for aggregates. |
| C-11 | Tag = constructor declaration order everywhere (host `List.mapi`, IR `state.variants`, C enum all iterate the declaration list). US-4 scenario 3 conformance check guards drift. |
| C-12 | US-3 gets dedicated unit snapshot tests (no e2e exists; creating one is optional at implement). |
| C-13, EC-8 | EVariant payload lowers positionally through the layout function (offsets = payload region offset + aligned cumulative of payload types — was: `4 + packed cumulative`) — same primitive as tuples, no ERecord rewrite. `_0/_1` is register-naming only, no byte-level counterpart needed. Host multi-field payload uses the same aligned cumulative offsets. |
| C-14, EC-10 | Separate `PtxLayout.v` with its own inductive; `PtxTypes.v` `elttype` untouched; existing theorems compile unmodified. |
| C-15 | Lemma set includes the natural-alignment side-condition over accepted layouts — the proof scope explicitly covers the alignment rule instead of silently implying safety. |
| C-16, EC-9 | Option (b): independent Rocq re-implementation + conformance test (exhaustive small-shape enumeration + qcheck random), run in dune tests. The host PPX's own aligned-layout functions (`aligned_record_offsets`/`variant_payload_offset`, PPX/AST-level — was: `calc_offsets`) stay; US-4 scenario 3 pins host agreement. |
| C-17 | DParam rejection message: names the param + "pass fields as separate scalar params or use a 1-element `t vector`" (1-element custom vectors are host-supported). |
| C-21 | ZLUDA execution IS the verification target: ACs run the 8 e2e tests under `LD_LIBRARY_PATH=~/opt/zluda` on the RX 7900 XTX and require per-device Status OK. |
| EC-1 | Whole-record element copy emits ALL field loads before ANY store (read-then-write), preventing intra-record RAW clobber under aliasing. |
| EC-2 | `Sarek_ir_layout` scalar sizes MUST equal host `field_byte_size` per type (source of truth; bool included), and scalar alignments MUST equal the host's natural alignment per type. Placement follows the C-2 aligned-layout rule; no scalar-size/alignment combination it produces is rejected as misaligned (was: any resulting misalignment falls under the C-2 rejection rule). |
| EC-6 | Payload bindings are lexically scoped to their arm; post-match code only sees pre-match registers. No merge machinery. |
| EC-11 | Discrimination is structural: `DParam (v, Some arr_info)` with aggregate `arr_elttype` = accepted vector-of-aggregate; bare `DParam (v, None)` with `TRecord/TVariant` var_type = rejected. |

## Functional Requirements

### Layout (US-1, US-2, US-4)
- **FR-001** [US-1]: The emitter MUST compute all aggregate byte offsets/sizes exclusively via `Sarek_ir_layout`; no offset arithmetic may be duplicated at emission sites.
- **FR-002** [US-1]: `Sarek_ir_layout` MUST reproduce the host **aligned (C-ABI)** record layout:
  each field is placed at the next offset satisfying its own natural alignment (padding
  inserted), the total record size is rounded up to the struct's maximum member alignment, and
  scalar sizes/alignments are identical to the host `field_byte_size`/natural-alignment mapping.
  Nested-record fields are flattened recursively on the same rule (`spoc/ir/Sarek_ir_layout.ml`
  `record_layout`; mirrored by the host PPX `sarek/ppx/Sarek_ppx.ml` `aligned_record_offsets`,
  lines 297-311, and `calc_type_size_early`, lines 313-315). *(Retracted 2026-07-30, campaign
  item L8 — was: "MUST reproduce the host packed layout: record offsets = cumulative
  `field_byte_size` sums; scalar sizes identical to host `field_byte_size`.")*
- **FR-003** [US-2]: Variant layout MUST be `[tag:int32 at 0][payload region at P]` where
  `P = max(4, max payload-member alignment)` over all constructors; each constructor's payload
  args are laid out from the start of the payload region on the same aligned-cumulative rule as
  a record; the element size is `round_up(P + max_payload_bytes, P)` (`4 + max_payload_bytes` is
  the special case where no payload member needs more than 4-byte alignment); tag = constructor
  declaration index (`spoc/ir/Sarek_ir_layout.ml` `variant_layout`; mirrored by the host PPX
  `variant_payload_offset`, lines 348-350, and `calc_variant_size`, lines 353-358). *(Retracted
  2026-07-30 — was: "Variant layout MUST be `[tag:int32 at 0][payload region at 4]`, element size
  `4 + max_payload_bytes` ... multi-arg payload offsets = 4 + packed cumulative.")*
- **FR-004** [US-5]: The layout function MUST reject (typed error) any aggregate containing a
  variant nested below top level, or any array/vector-typed field or payload member; the error
  MUST name the type and field/payload path and the reason (`Nested_variant` /
  `Unsupported_field`, `spoc/ir/Sarek_ir_layout.mli`). Under the aligned rule, no well-formed
  aggregate can place a scalar leaf at a non-naturally-aligned offset — `PtxLayout.v` proves this
  unconditionally (`record_always_accepted`, `variant_always_accepted`); the `Misaligned_field`
  error variant is retained only as a defensive internal invariant that MUST NOT fire for
  well-formed input. *(Retracted 2026-07-30 — was: "The layout function MUST reject (typed
  error) any aggregate placing a scalar leaf at a non-naturally-aligned offset; the error MUST
  name type, field, offset, and required alignment.")*
- **FR-005** [US-1]: Nested records MUST be supported by recursive flattening; aggregates containing variants below top level MUST be rejected with a precise error.

### Emission — global memory (US-1, US-2)
- **FR-010** [US-1]: Element addressing for aggregate vectors MUST use general byte-stride multiplication; scalar power-of-2 paths MUST remain shift-based (no regression).
- **FR-011** [US-1]: `ERecordField`/`LRecordField` on vector elements MUST load/store single fields at `base + idx*stride + offset` with the field's typed ld/st.
- **FR-012** [US-1]: Whole-aggregate element reads MUST materialize an SROA register set; whole-aggregate element writes MUST emit all loads before any store (EC-1).
- **FR-013** [US-2]: Variant element reads MUST load the tag and only the payload slots of the constructors the consuming match requires (or all slots — implementer's choice — but never read past `sizeof`).

### Emission — local values (US-1, US-2, US-3)
- **FR-020** [US-1]: Local record values MUST be SROA register sets; `ERecord` construction = per-field evaluation into fresh registers; `ERecordField` = register selection (no memory).
- **FR-021** [US-2]: Local variant values MUST be tag register + per-(ctor,arg) registers; `EVariant` sets the tag and its ctor's registers.
- **FR-022** [US-2]: `EMatch`/`SMatch` MUST lower to a tag-compare branch chain; EMatch MUST NOT use selp; payload bindings are arm-scoped; exhaustiveness per C-9 resolution.
- **FR-023** [US-1]: EApp inlining MUST accept aggregate-typed parameters (bind the caller's SROA set, copy scalars per existing copy_reg discipline applied per leaf) and aggregate returns (`inline_ret` carries a register set).
- **FR-024** [US-3]: `ETuple` MUST lower as an anonymous SROA aggregate with positional slots; tuple storage into global vectors MUST be rejected with a precise error.
- **FR-025** [US-1]: `SAssign (LArrayElem …, ERecord …)` MUST be supported (construct SROA then field-wise store, or direct per-field store — observable behavior per FR-012).

### Params & errors (US-5)
- **FR-030** [US-5]: `DParam` of bare `TRecord`/`TVariant` MUST be rejected with the C-17 message; `DParam` with `arr_info` whose `arr_elttype` is an accepted aggregate MUST be accepted and register its stride/layout for addressing.
- **FR-031** [US-5]: All new `unsupported` messages MUST name the construct and a workaround.

### Formal (US-4)
- **FR-040** [US-4]: A standalone `PtxLayout.v` MUST model the layout function without modifying `PtxTypes.v`'s `elttype`; the three existing theorems MUST still compile unmodified.
- **FR-041** [US-4]: Lemmas proved with 0 admits: field non-overlap; in-bounds; size correctness; natural alignment of every scalar leaf in accepted layouts.
- **FR-042** [US-4]: A conformance test MUST compare the Rocq layout model and the OCaml layout function on an exhaustive small-shape enumeration plus qcheck-random shapes, and MUST pin host agreement on the e2e test types (`point`, `point3d`, `color`, `particle`) via literal offset/size asserts. Per the plan's consensus decision (fact-checked: the extraction dune wiring is a stub), the Rocq side is represented by a hand-mirror OCaml transcription following the project's established CMBT pattern — not extracted code; the host pins are literal (a live `custom_type` get/set query would pull the PPX into the formal test's dependencies).

### Tests & integration (US-1..US-3)
- **FR-050** [US-1]: `generate_with_types` MUST become the real entry point consuming `kern_types`/`kern_variants`; plain `generate` behavior for scalar kernels MUST be unchanged (existing snapshot suite green).
- **FR-051** [US-1..3]: Unit snapshot tests MUST cover: record construct/field/global-roundtrip; nested record; non-pow2 stride; variant construct/match (nullary, 1-arg, multi-arg, ≥3 ctors); EMatch value position; tuple construct/project; helper with record arg+return; each rejection error of US-5.
- **FR-052** [US-1, US-2]: The 8 target e2e tests MUST pass with per-device Status OK on CUDA/PTX under ZLUDA (RX 7900 XTX); the 44 currently-passing e2e tests MUST remain passing (CPU + ZLUDA).
- **FR-053** [US-1]: `test_ktype_record`'s blanket non-Native SKIP MUST be removed per C-4.

## Acceptance Criteria

- AC-1 [US-1 happy path]: The 7 record e2e tests pass on the CUDA/PTX device under ZLUDA (per-device OK, epsilon-verified). ↔ CHECK-1
- AC-2 [US-2 happy path]: `test_registered_variant` passes on CUDA/PTX under ZLUDA. ↔ CHECK-2
- AC-3 [US-1, C-1]: PTX for a point3d copy kernel contains stride multiplication (no shl-only path) and three field ld/st pairs. ↔ CHECK-3
- AC-4 [US-2, C-10]: PTX for an EMatch kernel contains a tag branch chain and no `selp` for the match result. ↔ CHECK-4
- AC-5 [US-5, C-2]: Codegen of a mixed-alignment record and an f64-payload variant COMPILES,
  placing the misaligned-under-packed field/payload at its correct aligned offset; codegen of a
  variant nested below top level, or an array/vector-typed field or payload member, still fails
  with a precise error naming the type and field/payload path. ↔ CHECK-5 (was: "Codegen of a
  mixed-alignment record and an f64-payload variant fails with errors naming
  field/offset/alignment" — retracted 2026-07-30, campaign item L8.)
- AC-6 [US-4]: `PtxLayout.v` lemmas compile with 0 admits; conformance test green; existing 3 theorems and 30 CMBT tests unchanged-green. ↔ CHECK-6
- AC-7 [US-1..3, regression]: Full `dune runtest` green; 44-test ZLUDA e2e baseline preserved. ↔ CHECK-7
- AC-8 [US-3]: Snapshot tests for tuples + multi-arg payload pass. ↔ CHECK-8

## Runnable Checks

- CHECK-1 [AC-1]: `for t in ktype_record ktype_helper registered_type transpose nbody_ppx ray_ppx complex_types; do LD_LIBRARY_PATH=$HOME/opt/zluda _build/default/sarek/tests/e2e/test_$t.exe; done` → outputs contain no `generate_source returned None`, no per-device ERROR for CUDA/PTX, verifications OK.
- CHECK-2 [AC-2]: same for `test_registered_variant`.
- CHECK-3 [AC-3]: `opam exec -- dune exec sarek/tests/unit/test_ptx_snapshot.exe` — marker asserts for stride-mul + field pairs.
- CHECK-4 [AC-4]: snapshot marker asserts (branch labels + absence of selp on match result).
- CHECK-5 [AC-5]: `opam exec -- dune exec sarek/tests/unit/test_ptx_snapshot.exe` —
  `test_mixed_align_record_param_accepted` and `test_f64_variant_param_accepted` assert the
  aligned offsets/strides and that codegen succeeds; `opam exec -- dune exec
  sarek/tests/unit/test_ir_layout.exe` — `test_reject_variant_in_record` and
  `test_reject_vec_in_record` assert the still-rejected cases' exact error strings. (Was:
  "snapshot rejection tests assert the exact error strings" for both cases — retracted
  2026-07-30.)
- CHECK-6 [AC-6]: `opam exec -- dune build formal/ && opam exec -- dune runtest formal/` (0 admits enforced by build; conformance suite green).
- CHECK-7 [AC-7]: `opam exec -- dune runtest` + full ZLUDA e2e sweep script (intake Quality Gates).
- CHECK-8 [AC-8]: included in the snapshot suite run (CHECK-3 binary).

## Edge Cases (resolved dispositions)

- EC-1 → FR-012 (loads-before-stores). EC-2 → FR-002 + FR-004. EC-3 → FR-010.
- EC-4 → FR-022 (single post-typing result type). EC-5 → C-9 rule. EC-6 → arm-scoped bindings.
- EC-7 → FR-023. EC-8 → C-13 (positional aligned-cumulative offsets both sides — was: packed offsets). EC-9 → FR-042 domain.
- EC-10 → FR-040 (separate module). EC-11 → FR-030 (structural discrimination).

## Non-Goals (recorded)

- Record/variant kernel params by value from host (rejected with a precise error). (Was also:
  "mixed-alignment aggregates ... revisit if the host layout gains alignment" — the host layout
  gained alignment, campaign item L8, 2026-07-30; mixed-alignment aggregates are laid out
  correctly, not rejected, and are no longer a non-goal.)
- `pragma sarek.inline` bounded recursion; f64 transcendentals (separate gaps).
- Fixing the CUDA/OpenCL ERecord/nullary-EVariant reference divergences (flagged in intake).
- Full Rocq emitter re-proof over aggregates (follow-up task; boundary documented in PtxLayout.v).
