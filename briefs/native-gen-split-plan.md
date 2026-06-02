# Plan — native-gen-split

**Date:** 2026-06-02
**Status:** VALIDATED

## Sequential steps

1. **Create `Sarek_native_gen_base.ml`** with lines ~35–220 content (context, sets, name
   helpers, `gen_literal`, `gen_variable`, type-id helpers). SPDX header + `open` the same
   modules the original opens (check lines 19–31). Build `sarek/ppx/` — expect the original
   to now have duplicate defs; that's fixed in step 5.
2. **Create `Sarek_native_gen_expr.ml`** with the six sub-generators (lines ~222–695),
   `open Sarek_native_gen_base` (or qualify). SPDX header.
3. **Create `Sarek_native_gen_kernel.ml`** with lines ~1125–1728 (arg cast, types object,
   CPU kernel builders). Depends on base + expr + core entry points. SPDX header.
4. **Add the three new modules to `sarek/ppx/dune`** `(modules …)` list.
5. **Reduce `Sarek_native_gen.ml`** to the core: delete the moved sections; keep
   `gen_expr_impl`/`gen_pattern_impl`/`gen_binop`/`gen_unop`, entry points, and
   module/type-decl generation. Add `open` / qualification for base + expr. The recursive
   core calls the sub-generators as `Sarek_native_gen_expr.gen_memory_access …`.
6. **Update the single external caller** `sarek/ppx/Sarek_quote.ml:664`:
   `Sarek_native_gen.gen_cpu_kern_native_wrapper` →
   `Sarek_native_gen_kernel.gen_cpu_kern_native_wrapper`.
7. **Build + test + format + license** (gates below). Iterate until green.

## Dependencies

base ← expr ← core (`Sarek_native_gen`) ← kernel. No cycle. Build after each module to
isolate failures. The compiler is the oracle for completeness of the move.

## Identified risks

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| Subtle output change during move | Low | High (PPX compiles all kernels) | Pure copy; `@sarek/tests/runtest` native codegen tests gate it; no logic edits |
| Accidentally "fixing" a documented bug | Low | High | Brief forbids it; reviewer checks the four bug sites are moved verbatim |
| Missed reference / forward dep | Med | Low (compile error) | Incremental per-module build |
| Shared helper used by both expr and kernel placed in wrong module | Med | Low (compile error) | base holds all shared leaf helpers |

## Decisions made

| Point | Decision | Reason |
|---|---|---|
| Single caller fix | Update `Sarek_quote.ml` call site (not re-export) | Re-export from core would create core→kernel cycle |
| Dual-voice planning | Skipped | Design fully determined by code reading; no scope ambiguity |
| Documented bugs | Move verbatim, do NOT fix | Each deserves its own intake; mixing is untestable |
| `.mli` files | None (match existing) | No `.mli` exists for the original; not required |

## Assumptions

- Only one external caller exists (`Sarek_quote.ml:664`) — confirmed by grep.
- All sub-generators are `~gen_expr`-parameterised (confirmed: no direct `gen_expr_impl`
  call inside them).
- Native codegen is exercised by `@sarek/tests/runtest` (per `kb/sarek/ppx/native-gen.md`
  related tests).
