# Status — codegen-ptx formal project

_Last reconciled against the tree: 2026-07-30. Every count below is taken from
`proof-ledger.json`, which is **generated**, not hand-written:
`scripts/check-formal-proofs.sh` rebuilds all `.v` from scratch inside
`rocq/rocq-prover:9.1.1`, regenerates the ledger via
`scripts/gen-proof-ledger.py`, and fails CI if the committed ledger differs. If
this file and the ledger ever disagree, the ledger is right._

Track B steps 10-12 complete.

**Theories: 6 files compiling** — AGpuSemantics, PtxTypes, PtxExprSpec,
PtxStmtSpec, PtxKernelSpec, PtxLayout.

| Metric | Value | Where it comes from |
|---|---|---|
| Theorems | **79** | `proof-ledger.json` → `counts.theorems` |
| Admits | 0 | no `Admitted`/`admit` under `theories/`; `check-formal-proofs.sh` greps for them *and* the Rocq build would fail anyway |
| Project-local axioms / `Parameter`s | 6 | the f32/f64 `sin`/`cos`/`fma` of `AGpuSemantics.v`, all six sanctioned in `formal/axiom-allowlist.txt` |
| Toolchain-base axioms (inherited) | 99 | `counts.axioms_toolchain_base` — Rocq's own primitive-float/int63 axiomatisation, not ours to sanction |

Per module (`counts.theorems`): PtxLayout 50, PtxTypes 16, AGpuSemantics 5,
PtxExprSpec 4, PtxKernelSpec 2, PtxStmtSpec 2.

The axiom allowlist is enforced in **both** directions — an axiom the kernel
found that is not listed fails, and a listed name no proof depends on any more
also fails — so the six cannot quietly become seven, nor stay listed after
becoming unused.

## Headline theorems

1. `emit_expr_correct` (PtxExprSpec.v):
   `agpu_eval_ir st e = Some (v, st') → agpu_eval_ptx st (emit_ast_expr e) = Some (v, st')`

2. `emit_stmt_correct` (PtxStmtSpec.v):
   `agpu_exec_ir st s = agpu_exec_ptx_stmt st (emit_ast_stmt s)`
   via auxiliary `eval_ir_ptx_eq`:
   `agpu_eval_ir st e = agpu_eval_ptx st (emit_ast_expr e)`

3. `emit_kernel_correct` (PtxKernelSpec.v):
   `agpu_exec_ir_kernel st k = agpu_exec_ptx_kernel st (emit_ast_kernel k)`

Plus `emit_ast_kernel_shared_preserved` (PtxKernelSpec.v) and the PtxLayout
theorems listed below. The full per-module theorem list is in
`proof-ledger.json`.

## Extraction — and what it is actually wired to

There are two extraction entry points, and they are **not** equivalent in
status. Conflating them is what let this file previously read as though the
whole model were extracted.

- **`extraction/LayoutExtract.v` → `extraction/sarek_ptx_layout_model.ml{,i}`.**
  Real, linked, and CI-drift-checked. One self-contained pair of files
  (`Extraction "file.ml"`, every dependency inlined) that dune builds and
  `test/test_layout_conformance.ml` links as its `Model`.
  `scripts/check-formal-proofs.sh` re-extracts from scratch, runs
  `scripts/canonicalize-extraction.py` on both sides (the Rocq pretty-printer's
  line breaking depends on which OCaml the *prover* was built with, so a raw
  byte-compare would fire on every run), and byte-compares against the
  committed copies — a `PtxLayout.v` edit that is not propagated fails CI.
- **`extraction/Extract.v` → five `.ml` files in the project root**
  (AGpuSemantics, PtxTypes, PtxExprSpec, PtxStmtSpec, PtxKernelSpec). These use
  `Extraction Library`, which emits one file per Rocq module still referring to
  its dependencies by name (`open Datatypes`, `open List`, `open Nat`) —
  modules that exist only if the Stdlib is extracted alongside. **They have
  never compiled, and nothing links them.** They are committed and regenerated
  by every proof run, and that is all. Float `Parameter`s are mapped to
  `Stdlib.Float` wrappers there in anticipation, not in use.

So the emitter's expression/statement/kernel model is **conformance-tested
against a hand-written mirror**, and only the layout model is checked against
Rocq's own extraction. That asymmetry is the honest answer to
"model vs production" for this project.

## Conformance tests

### Expression/statement/kernel suite — hand-mirrored

`test/test_codegen_ptx_conformance.ml`: **37** Alcotest cases across nine
groups (literals 7, thread-intrinsics 4, arithmetic 4, comparisons 4,
math-intrinsics 6, type-safety 2, registers 2, barrier 1, ptx-dshared 7).

Both sides of every `eval_agrees` check use the OCaml mirror defined in that
file, not the extracted Rocq modules — see the extraction section above for
why. It validates the mirror's internal consistency with the spec; it does not
close the mirror↔theory hop.

### Layout suite — runs the extracted theory

`test/test_layout_conformance.ml` (Alcotest + qcheck-core seeded generator):
**11** cases in five groups, checking the **extracted** `PtxLayout.v` against
`Sarek_ir_layout` for accept/reject and all offsets/sizes:

- exhaustive records ≤4 fields over {i32,f32,bool,i64,f64} (780 shapes);
- exhaustive variants ≤3 ctors × ≤2 args (30 783 shapes);
- 500 seeded random nested records (depth ≤3);
- host pins for `point`, `point3d`, `color`, `particle`;
- mixed-alignment pins from the aligned-ABI migration: `{i32;f64}` a@0 b@8
  size 16, `{bool;f64}` d@8 size 16, reordered `{f64;i32}` b@0 a@8 size 16,
  and a variant with an f64 payload at offset 8, size 16.

> **Historical record — the `RocqMirror` hop is closed.** `Model` used to be a
> ~130-line hand transcription of `PtxLayout.v` living inside this test file,
> described in its own header as "a line-by-line OCaml transcription". Every
> theorem was proved about the Rocq definitions and then checked against a copy
> of them that no tool compared to the original, so a theory edit nobody
> propagated left the suite green while it tested a model that had stopped
> being the model. The 2026-07-24 audit filed this as "2 unlinked
> human-maintained hops OCaml ↔ mirror ↔ proof". Extraction removed the hop
> rather than watching it. Kept here as rationale, not as a task — do not
> reintroduce a mirror for the layout model.

## Layout model (PtxLayout.v, FR-040/041/042)

`theories/PtxLayout.v` (registered in `_CoqProject`, builds via `make -f
CoqMakefile`): standalone **aligned (C-ABI)** aggregate-layout model of
`spoc/ir/Sarek_ir_layout.ml`. **50 theorems, 0 Admitted.**

> **Retraction — this model is no longer packed.** Earlier revisions of this
> file described it as "packed cumulative offsets, no padding". That was true of
> the original model and is now false: campaign item L8 migrated the host PPX,
> `Sarek_ir_layout` and this theory from PACKED to ALIGNED. Record fields are
> placed at the next offset satisfying their natural alignment (padding
> inserted), total size is rounded up to the struct's maximum member alignment,
> and a variant is `[tag:int32@0][payload@P]` with `P = max(4, max payload
> member alignment)`, size rounded to the overall alignment. That is the
> standard C struct ABI, so it agrees byte-for-byte with the `typedef struct
> {...}` the C-family backends emit. The consequences are recorded in the
> theory's own header and reflected in the theorem names below; the point of
> retracting in place is that the old description is the one a reader would
> otherwise carry into an ABI question.

Structural facts that survived the migration: the model deliberately does not
touch `PtxTypes.elttype` (it has its own two-point scalar universe `lty`
carrying byte size and natural alignment only, field names as `nat` indices);
"no variant below top level" is structural, because `lfield` has no variant
constructor at all. PtxLayout declares no axiom of its own and imports only
`Stdlib`, so it does not rest on the six math `Parameter`s either;
`check-formal-proofs.sh` runs `coqchk` over the compiled theories and
`gen-proof-ledger.py` reads the kernel's own context summary, so a new
dependency would appear as ledger drift rather than as prose here going stale.

What the migration did to the invariants — the load-bearing part:

- The old master invariant `chain` said leaves **tile** the byte range with no
  gaps (`off' = off + leaf_size`). Aligned layout inserts padding, so it is
  false. It is replaced by `sorted_packed` (leaves ordered and non-overlapping,
  gaps permitted) plus `end_of` (the running end).
- The old `record_size_correct` ("size = sum of leaf sizes") is likewise false
  with padding; it is restated as `record_size_is_padded_end` (size = the
  aligned, padded cumulative end).
- Non-overlap, in-bounds and alignment survive — and alignment became
  *unconditional* rather than conditional on acceptance, which is why
  `record_always_accepted` / `variant_always_accepted` hold and the OCaml
  `Misaligned_field` rejection is now dead code on the record path.

Key theorems (see `proof-ledger.json` for all 50):

- alignment arithmetic — `align_up_ge`, `align_up_add`, `align_up_divide`,
  `divide_mod0`, `scalar_align_pos`, `falign_pos`, `fsalign_pos`.
- ordering / non-overlap — `sorted_app`, `sorted_lower`, `sorted_nth_le`,
  `sorted_weaken`, `flattens_sorted`, `record_sorted`,
  `record_leaf_nonoverlap`, `variant_ctor_leaf_nonoverlap`,
  `variant_tag_payload_disjoint`.
- in-bounds — `record_leaf_in_bounds`, `variant_leaf_in_bounds`, `in_le_end`,
  `end_of_ge`, `flattens_end`.
- size — `record_size_is_padded_end`, `ctor_payload_size_correct`,
  `max_payload_ub`, `variant_payload_offset_*`.
- alignment of placed leaves — `record_leaf_aligned`, `variant_leaf_aligned`,
  `flattens_aligned`, `flatten_aligned_mut`.
- acceptance — `record_always_accepted`, `variant_always_accepted`.
- tag discipline — `ctor_tag_is_index` (stored tag of the i-th declared
  constructor is i), `ctor_layouts_spec`.

## Key design decisions

- `ptx_intrinsic_tag` split into type-specific variants (PISin32/PISin64 etc.)
  to ensure `eval_ir_ptx_eq` holds without a typing predicate.
- `IEArrayRead` restricted to uniform types (U32+U32 or U64+U64) to match
  PTX's type-homogeneous address arithmetic.
- `F64 Le` bug in `agpu_eval_binop` corrected (`leb a b`, not `leb b a`).
- The six math intrinsics stay **uninterpreted**. The abstract GPU semantics
  reasons about how a kernel *uses* `sin`/`cos`/`fma`, never about their numeric
  results, so axiomatising them keeps the model independent of any device's ULP
  behaviour; interpreting them would commit the spec to one hardware rounding
  story and prove nothing the codegen theorems need. Rationale is also recorded
  in `formal/axiom-allowlist.txt`, where it is enforced.
