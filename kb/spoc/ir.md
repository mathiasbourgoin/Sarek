# spoc/ir

## Component Inventory

- `spoc/ir/README.md`: overview of the GPU kernel IR hierarchy, APIs, analysis, and testing.
- `spoc/ir/dune`: builds public library `spoc.ir` as unwrapped modules `Sarek_ir_types`, `Sarek_ir_pp`, `Sarek_ir_analysis`, and `Sarek_ir_codegen` (`spoc/ir/dune:4-11`).
- `spoc/ir/Sarek_ir_types.ml`: pure IR type definitions plus native execution helper types.
- `spoc/ir/Sarek_ir_pp.ml`: string conversion and `Format` pretty-printers for IR nodes.
- `spoc/ir/Sarek_ir_analysis.ml`: float64 usage analysis across types, expressions, statements, declarations, helpers, and kernels.
- `spoc/ir/Sarek_ir_codegen.ml` (+ `.mli`): shared GPU code-generation helpers extracted from the backends to avoid duplicated variant/struct emission — `mangle_name`, `gen_variant_def` (C/MSL tagged-union variant emission for CUDA/OpenCL/Metal), and `gen_variant_def_glsl` (GLSL variant emission for Vulkan).
- `spoc/ir/test/*`: construction, printing, and float64 analysis tests.

## Per-File Purpose

- `Sarek_ir_types.ml` defines memory spaces, element types, variables, constants, operators, expressions, lvalues, statements, declarations, helper functions, native arguments, native closures, native functions, and kernels (`spoc/ir/Sarek_ir_types.ml:11-245`).
- `Sarek_ir_pp.ml` renders IR to debug/source-like strings (`spoc/ir/Sarek_ir_pp.ml:10-249`).
- `Sarek_ir_analysis.ml` recursively detects `TFloat64` and `CFloat64` use (`spoc/ir/Sarek_ir_analysis.ml:10-111`).
- `Sarek_ir_codegen.ml` emits backend-agnostic variant types: `mangle_name` normalizes type names into valid C/GLSL identifiers (`spoc/ir/Sarek_ir_codegen.mli:12`), `gen_variant_def` emits an enum + tagged-union struct + per-case constructor functions parameterised by `type_of_elttype` and `constructor_prefix` (`spoc/ir/Sarek_ir_codegen.mli:24-29`), and `gen_variant_def_glsl` emits the GLSL equivalent without enum/typedef/union (`spoc/ir/Sarek_ir_codegen.mli:38-42`).

## Features and APIs

- Element types cover scalar primitives, records, variants, arrays with memory space, and vectors (`spoc/ir/Sarek_ir_types.ml:15-28`).
- Expressions support constants, variables, binary/unary ops, array access, record fields, intrinsics, casts, tuples, apps, records, variants, array length/create, value `if`, and value `match` (`spoc/ir/Sarek_ir_types.ml:79-101`).
- Statements support assignments, sequences, conditionals, loops, matches, returns, barriers, memory fences, let bindings, pragmas, scoped blocks, and native snippets (`spoc/ir/Sarek_ir_types.ml:110-132`).
- Native vector helpers wrap `Obj.magic` access behind `vec_get_custom`, `vec_set_custom`, `vec_length`, and `vec_as_vector` (`spoc/ir/Sarek_ir_types.ml:190-218`).
- `pp_kernel` prints a kernel header, params, locals, and body (`spoc/ir/Sarek_ir_pp.ml:236-247`).
- `kernel_uses_float64` combines parameter/local/body/helper/type/variant checks (`spoc/ir/Sarek_ir_analysis.ml:96-111`).

## Invariants

- `var_id` is intended to be unique for alpha-renaming, while `var_name` remains human-readable (`spoc/ir/README.md:66-72`).
- `DParam` array information should agree with vector/array parameter shape (`spoc/ir/Sarek_ir_types.ml:134-142`).
- `SLet` uses immutable variables and `SLetMut` uses mutable variables by convention (`spoc/ir/Sarek_ir_types.ml:123-124`).
- `NativeFn` receives typed `native_arg array` plus block/grid dimensions and a `parallel` flag (`spoc/ir/Sarek_ir_types.ml:220-227`).
- Float64 analysis should conservatively return true if any reachable type or expression requires double precision.

## Potential Invariant Violations or Bugs

- Lvalue types are not inspected in `stmt_uses_float64`; `SAssign` checks only the right-hand expression (`spoc/ir/Sarek_ir_analysis.ml:53-56`). A statement assigning a non-float64 expression into a `TFloat64` variable could be missed unless the variable is declared elsewhere in the kernel. This matters for isolated `stmt_uses_float64` callers.
- `SNative` is treated as not using float64 (`spoc/ir/Sarek_ir_analysis.ml:72-73`). Marked uncertain because native code may be intentionally opaque, but this can under-report requirements.
- `EMatch` pretty-printing discards actual expression patterns and prints `_` for each case (`spoc/ir/Sarek_ir_pp.ml:111-116`).
- `SFor` pretty-printing uses `<` for `Upto` and `>` for `Downto` (`spoc/ir/Sarek_ir_pp.ml:146-162`), which excludes the stop expression. OCaml-style `to`/`downto` loops are usually inclusive, so the IR semantics need clarification.
- `pp_kernel` does not print `kern_types`, `kern_variants`, `kern_funcs`, or `kern_native_fn` (`spoc/ir/Sarek_ir_pp.ml:236-247`).
- Native helpers use `Obj.magic` (`spoc/ir/Sarek_ir_types.ml:191-202`, `214-218`), so type safety depends on PPX/runtime construction discipline.

## Performance and Maintainability Risks

- Recursive analysis and pretty-printing are straightforward but not tail-recursive for deeply nested expressions/statements.
- No validator enforces variable uniqueness, declaration consistency, field existence, constructor arity, array memory spaces, or statement expression types.
- Debug pretty-printing resembles C/CUDA syntax in places but is incomplete; maintainers could accidentally rely on it for code generation.
- Unwrapped modules (`spoc/ir/dune:7`) ease access but increase namespace collision risk.

## Related Tests

- `spoc/ir/test/test_sarek_ir_types.ml` covers construction of most variants and helpers.
- `spoc/ir/test/test_sarek_ir_pp.ml` covers primitive string conversions and representative expression/statement/declaration/kernel printing.
- `spoc/ir/test/test_sarek_ir_analysis.ml` covers float64 detection across many type, expression, statement, declaration, helper, and kernel shapes.

## Missing Tests

- `SAssign` where the lvalue type is `TFloat64` and expression is not.
- `SNative` analysis policy.
- Pretty-printing for `EArrayReadExpr`, `EMatch` patterns, `SMatch`, `SWhile`, `SFor Downto`, `SLetMut`, `SPragma`, `SBlock`, `SNative`, helper functions, record/variant definitions, and empty/non-empty tuple corner cases.
- Native helper failure paths (`vec_get_custom`, `vec_set_custom`, `vec_as_vector` on non-`NA_Vec`).
- IR structural validation and semantic invariants.

## Concrete Improvement Candidates

- Add `lvalue_uses_float64` and include it in `SAssign` analysis.
- Define and document exact `SFor` bound semantics; adjust pretty-printer/tests if inclusive loops are intended.
- Make `SNative` carry metadata such as required capabilities or `uses_float64`.
- Add an `Sarek_ir_validate` module returning structured errors for malformed IR.
- Rename or document `Sarek_ir_pp` as debug-only if it must not be used as source generation.
