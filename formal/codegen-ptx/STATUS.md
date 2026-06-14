# Status — codegen-ptx formal project
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

## Key design decisions

- `ptx_intrinsic_tag` split into type-specific variants (PISin32/PISin64 etc.)
  to ensure `eval_ir_ptx_eq` holds without a typing predicate.
- `IEArrayRead` restricted to uniform types (U32+U32 or U64+U64) to match
  PTX's type-homogeneous address arithmetic.
- `F64 Le` bug in `agpu_eval_binop` corrected (`leb a b`, not `leb b a`).

