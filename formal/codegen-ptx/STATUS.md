# Status — codegen-ptx formal project

Track B steps 10-12 complete.

Theories: 5 files compiling (AGpuSemantics, PtxTypes, PtxExprSpec, PtxStmtSpec, PtxKernelSpec).
Theorems proved: 3. Admits: 0.

## Theorems

1. `emit_expr_correct` (PtxExprSpec.v):
   `agpu_eval_ir st e = Some (v, st') → agpu_eval_ptx st (emit_ast_expr e) = Some (v, st')`

2. `emit_stmt_correct` (PtxStmtSpec.v):
   `agpu_exec_ir st s = agpu_exec_ptx_stmt st (emit_ast_stmt s)`
   via auxiliary `eval_ir_ptx_eq`:
   `agpu_eval_ir st e = agpu_eval_ptx st (emit_ast_expr e)`

3. `emit_kernel_correct` (PtxKernelSpec.v):
   `agpu_exec_ir_kernel st k = agpu_exec_ptx_kernel st (emit_ast_kernel k)`

## Extraction

`extraction/Extract.v` extracts all 5 modules to OCaml. Extracted `.ml` files
appear in the project root. Float parameters mapped via `ExtrOCamlFloats`.

## Conformance tests

`test/test_codegen_ptx_conformance.ml`: 30 Alcotest CMBT smoke tests, all passing.
Tests cover: literals, thread intrinsics, arithmetic, comparisons, math intrinsics,
type-safety (wrong-type → None), register reads, barrier.

## Layout model (PtxLayout.v, FR-040/041/042)

`theories/PtxLayout.v` (registered in `_CoqProject`, builds via `make -f
CoqMakefile`): standalone packed-layout model of `spoc/ir/Sarek_ir_layout.ml`
(records: packed cumulative offsets, no padding; variants:
`[tag:int32@0][payload@4]`, size `4 + max payload`, tag = declaration index).
Does not touch `PtxTypes.elttype`; "no variant below top level" is structural
(`lfield` has no variant constructor). 22 Qed, 0 Admitted; every theorem's
`Print Assumptions` reports closed under the global context.

Lemmas/theorems:

- `scalar_size_pos` — every scalar has positive byte size.
- `leaves_size_app` / `chain_app` — leaf-size and packed-chain concatenation.
- `flatten_chain_size` (+ `flattens_chain`, `flattens_leaves_size`) — master
  invariant: flattening from offset `off` tiles bytes consecutively with total
  leaf size equal to the packed size (no padding, no gaps).
- `chain_in_lower` / `chain_in_bounds` / `chain_nth_lower` /
  `chain_nonoverlap` — consequences: every leaf starts at/after the chain
  origin, ends within the chain extent, and earlier leaves end before later
  ones begin.
- `record_size_correct` — record size = sum of scalar leaf sizes.
- `record_leaf_in_bounds` — every record leaf's byte range fits in the record.
- `record_leaf_nonoverlap` — distinct record leaves occupy disjoint ranges.
- `record_accepted_aligned` — accepted layouts place every leaf at a multiple
  of its natural alignment.
- `ctor_layouts_ok` / `max_payload_ub` — per-constructor chain/size facts and
  the max-payload upper bound.
- `ctor_tag_is_index` — stored tag of the i-th declared constructor is i.
- `variant_tag_payload_disjoint` — tag bytes `[0,4)` never overlap payload.
- `variant_ctor_leaf_nonoverlap` — payload leaves of one constructor are
  disjoint.
- `ctor_payload_size_correct` — recorded payload size = sum of leaf sizes.
- `variant_leaf_in_bounds` — every payload leaf fits in `4 + max payload`.
- `variant_accepted_aligned` — accepted variant payload leaves are naturally
  aligned.

### Layout conformance suite

`test/test_layout_conformance.ml` (Alcotest + qcheck-core seeded generator):
hand-mirror `RocqMirror` of PtxLayout.v checked against `Sarek_ir_layout` for
accept/reject and all offsets/sizes. 7 Alcotest cases, all passing:
exhaustive records ≤4 fields over {i32,f32,bool,i64,f64} (780 shapes),
exhaustive variants ≤3 ctors × ≤2 args (30 783 shapes), 500 seeded random
nested records (depth ≤3), and literal host pins for `point` (0,4/8),
`point3d` (0,4,8/12), `color` (tag@0, payload@4, size 8, decl-order tags),
`particle` (0..16/20). No mirror/OCaml divergence found.

## Key design decisions

- `ptx_intrinsic_tag` split into type-specific variants (PISin32/PISin64 etc.)
  to ensure `eval_ir_ptx_eq` holds without a typing predicate.
- `IEArrayRead` restricted to uniform types (U32+U32 or U64+U64) to match
  PTX's type-homogeneous address arithmetic.
- `F64 Le` bug in `agpu_eval_binop` corrected (`leb a b`, not `leb b a`).

