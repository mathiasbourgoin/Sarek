# Sarek Native Code Generation

## Component Inventory

Native fallback generation uses `sarek/ppx/Sarek_native_helpers.ml`, `sarek/ppx/Sarek_native_intrinsics.ml`, and the `Sarek_native_gen*` module family. The former monolithic `Sarek_native_gen.ml` (~1951 lines) was split (pure code move) into four modules: `Sarek_native_gen_base.ml`, `Sarek_native_gen_expr.ml`, `Sarek_native_gen.ml` (reduced), and `Sarek_native_gen_kernel.ml`.

## Per-File Purpose

- `Sarek_native_helpers.ml`: converts source locations, builds stable variable names, maps variable IDs, and generates default values.
- `Sarek_native_intrinsics.ml`: maps Sarek types and intrinsic constants/functions to OCaml AST for native execution modes.
- `Sarek_native_gen_base.ml`: shared context/types, name helpers, and the leaf generators `gen_literal` (`sarek/ppx/Sarek_native_gen_base.ml:77`) and `gen_variable` (`sarek/ppx/Sarek_native_gen_base.ml:102`).
- `Sarek_native_gen_expr.ml`: the `~gen_expr`-parameterised sub-generators — memory access (`gen_memory_access`, `sarek/ppx/Sarek_native_gen_expr.ml:15`), let bindings (`gen_let_binding`, `:129`), control flow (`gen_control_flow`, `:159`), data structures (`gen_data_structure`, `:219`), special expressions (`gen_special_expr`, `:338`), and BSP.
- `Sarek_native_gen.ml` (reduced): the recursive core `gen_expr_impl` plus `gen_pattern_impl`/`gen_binop`/`gen_unop` (`sarek/ppx/Sarek_native_gen.ml:37-274`), public entry points (`gen_expr`, `gen_expr_with_inline_types`, `sarek/ppx/Sarek_native_gen.ml:281-290`), and module/type-declaration generation (`sarek/ppx/Sarek_native_gen.ml:292-462`).
- `Sarek_native_gen_kernel.ml`: argument casting (`gen_arg_cast`, `sarek/ppx/Sarek_native_gen_kernel.ml:32`), the types object (`gen_types_object`, `:217`), and CPU kernel builders (`gen_cpu_kern_native`, `:358`; `gen_simple_cpu_kern_native`, `:460`; `gen_cpu_kern_native_wrapper`, `:591`).

## Features And APIs

- Type-to-OCaml conversion is in `sarek/ppx/Sarek_native_intrinsics.ml:35-83`.
- Native intrinsic constants are generated in `sarek/ppx/Sarek_native_intrinsics.ml:121-219`.
- Native intrinsic functions are generated in `sarek/ppx/Sarek_native_intrinsics.ml:223-278`.
- Vector and local array get/set generation is in `sarek/ppx/Sarek_native_gen_expr.ml:15-126` (`gen_memory_access`).
- Record field access/set generation, including external qualified types and inline FCM, is in `sarek/ppx/Sarek_native_gen_expr.ml:15-126` (`gen_memory_access`) and the data-structure constructors in `sarek/ppx/Sarek_native_gen_expr.ml:219-335` (`gen_data_structure`).
- Loop generation is in `sarek/ppx/Sarek_native_gen_expr.ml:159-216` (`gen_control_flow`, `TEFor` at `:170`).
- Full CPU native kernels are built in `sarek/ppx/Sarek_native_gen_kernel.ml:358-459` (`gen_cpu_kern_native`).
- Simple native kernels are built in `sarek/ppx/Sarek_native_gen_kernel.ml:460-590` (`gen_simple_cpu_kern_native`).
- Final native wrappers and FCM handling are in `sarek/ppx/Sarek_native_gen_kernel.ml:591-842` (`gen_cpu_kern_native_wrapper`).

## Invariants

- Native fallback semantics must match Sarek IR/runtime semantics.
- Kernel integer indices are generally `int32` in Sarek but host loops and OCaml arrays often require `int`; conversions must be explicit.
- Simple/native execution mode must support all intrinsics that convergence analysis permits in that mode.
- Default values must exist for types used in mutable native temporaries and tailrec transformations.

## Potential Invariant Violations Or Bugs

- Confirmed: `downto` loop generation reverses bounds in `sarek/ppx/Sarek_native_gen_expr.ml:170-208` (`TEFor`); for source `for i = lo downto hi`, the branch emits an OCaml loop from `hi downto lo` (`sarek/ppx/Sarek_native_gen_expr.ml:205`).
- Confirmed: `TECreateArray` emits `Array.make [%e size_e]` in `sarek/ppx/Sarek_native_gen_expr.ml:331-334`; typed sizes are int32, while `Array.make` expects host `int`.
- Probable: `global_size_*` can be classified as simple execution but native simple modes reject many such constants in `sarek/ppx/Sarek_native_intrinsics.ml:191-218`.
- Confirmed limitation: custom scalar defaults fail at runtime in `sarek/ppx/Sarek_native_helpers.ml:82-124` unless the type has a record/variant representation.
- Confirmed limitation: mutable fields with inline FCM are explicitly unsupported around `sarek/ppx/Sarek_native_gen_expr.ml:101-102`.
- Confirmed limitation: custom scalar argument casting falls through to `failwith` in `sarek/ppx/Sarek_native_gen_kernel.ml:209-210`.

## Performance Or Maintainability Risks

- Native generation encodes a second semantics for loops, casts, records, intrinsics, memory, and execution grids.
- Full CPU mode uses nested host loops and generated closures; AST size can grow significantly after monomorphization/inlining.
- Simple execution mode has separate intrinsic support, so new built-ins need updates in both full and simple branches.

## Related Tests

- `sarek/tests/unit/test_native_helpers.ml:200-263` covers helper conversion, variable names, and default values.
- `sarek/tests/unit/test_native_intrinsics.ml:186-227` covers type conversion and selected intrinsic constants/functions.
- `sarek/tests/e2e/test_debug_native.ml` is built as an executable in `sarek/tests/e2e/dune:183-188`.
- `sarek/tests/new_runtime/test_native_runtime.ml` is declared in `sarek/tests/new_runtime/dune:4-9`.

## Missing Tests

- Native `downto` loop with nontrivial bounds.
- Native `create_array` from an int32 size.
- Simple native kernels using `global_size_x/y/z`.
- Custom scalar defaults and custom scalar argument casting.
- Mutable inline-FCM record field set behavior or explicit rejection.

## Concrete Improvement/Fix Candidates

- Fix `downto` generation to preserve source lower/upper expression order.
- Convert local array sizes with `Int32.to_int` consistently.
- Add a mode capability table for intrinsic constants/functions and use it in convergence and native generation.
- Replace native `failwith` paths for unsupported custom scalars with compile-time PPX errors where possible.
