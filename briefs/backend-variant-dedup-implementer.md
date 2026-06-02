# Implementer Brief — backend-variant-dedup

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

Extract `mangle_name` (4 identical copies) and `gen_variant_def` (3 C-style + 1 GLSL-style)
into a new shared module `Sarek_ir_codegen` in `spoc/ir/`. All four backends then call
the shared versions. No functional change to generated GPU code.

## Files to modify

| File | Change |
|---|---|
| `spoc/ir/dune` | Add `Sarek_ir_codegen` to `(modules ...)` |
| `spoc/ir/Sarek_ir_codegen.ml` | Create (new file) |
| `spoc/ir/Sarek_ir_codegen.mli` | Create (new file) |
| `sarek-cuda/Sarek_ir_cuda.ml` | Delete local `mangle_name`+`gen_variant_def`; update call sites |
| `sarek-opencl/Sarek_ir_opencl.ml` | Delete local `mangle_name`+`gen_variant_def`; update call sites |
| `sarek-metal/Sarek_ir_metal.ml` | Delete local `mangle_name`+`gen_variant_def`; update call sites; resolve dead-code |
| `sarek-vulkan/Sarek_ir_glsl.ml` | Delete local `mangle_name`+`gen_variant_def`; update call sites |

**Do NOT touch:** `generate_with_types`, `type_of_elttype` functions, test files
(unless the Metal dead-code decision requires adding a call in `generate_with_types`).

## Step-by-step

### Step 1 — Update `spoc/ir/dune`

Change:
```
(modules Sarek_ir_types Sarek_ir_pp Sarek_ir_analysis)
```
to:
```
(modules Sarek_ir_types Sarek_ir_pp Sarek_ir_analysis Sarek_ir_codegen)
```

### Step 2 — Create `spoc/ir/Sarek_ir_codegen.mli`

```ocaml
(* SPDX-License-Identifier: CECILL-B *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)

val mangle_name : string -> string
(** Replace '.' with '_' for C/GLSL identifier compatibility. *)

val gen_variant_def :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  constructor_prefix:string ->
  Buffer.t ->
  string * (string * Sarek_ir_types.elttype list) list ->
  unit
(** Emit C/MSL enum + typedef struct + union + inline constructor functions.
    Used by CUDA, OpenCL, Metal backends. *)

val gen_variant_def_glsl :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  Buffer.t ->
  string * (string * Sarek_ir_types.elttype list) list ->
  unit
(** Emit GLSL const-int enum + struct (flat fields, no union) + constructor functions.
    Used by Vulkan backend. *)
```

### Step 3 — Create `spoc/ir/Sarek_ir_codegen.ml`

Add SPDX header. Implement `mangle_name` as the shared one-liner.

For `gen_variant_def`: copy the body verbatim from `sarek-cuda/Sarek_ir_cuda.ml:796–868`,
replace every `cuda_type_of_elttype ty` with `type_of_elttype ty`, replace the prefix
string literal `"__device__ __host__ inline"` with `constructor_prefix`.

For `gen_variant_def_glsl`: copy the body verbatim from
`sarek-vulkan/Sarek_ir_glsl.ml:969–1036`, replace every `glsl_type_of_elttype ty`
with `type_of_elttype ty`. The GLSL version uses `Printf.sprintf` heavily — do not
switch to `Buffer.add_string` style; keep it as-is.

Verify: `opam exec --switch=/home/mathias/dev/SPOC -- dune build` passes.

### Step 4 — Migrate CUDA (`sarek-cuda/Sarek_ir_cuda.ml`)

1. Delete lines 33–34 (local `mangle_name`).
2. Delete lines 796–868 (local `gen_variant_def`).
3. Replace all remaining `mangle_name` call sites with `Sarek_ir_codegen.mangle_name`.
   Call sites to update: lines 43, 44, 148, 797 (now gone), 896 in `generate_with_types`.
   Search: `grep -n 'mangle_name' sarek-cuda/Sarek_ir_cuda.ml`
4. Update `generate_with_types` line 881 (renumbered after deletions):
   ```ocaml
   List.iter
     (Sarek_ir_codegen.gen_variant_def
        ~type_of_elttype:cuda_type_of_elttype
        ~constructor_prefix:"__device__ __host__ inline"
        buf)
     k.kern_variants ;
   ```
5. `opam exec --switch=/home/mathias/dev/SPOC -- dune build` — must be clean.

### Step 5 — Migrate OpenCL (`sarek-opencl/Sarek_ir_opencl.ml`)

Same pattern. Local `mangle_name` at line 44. Local `gen_variant_def` at line 778.
Call site in `generate_with_types` at line 859.

```ocaml
List.iter
  (Sarek_ir_codegen.gen_variant_def
     ~type_of_elttype:opencl_type_of_elttype
     ~constructor_prefix:"static inline"
     buf)
  k.kern_variants ;
```

`dune build` after.

### Step 6 — Migrate Metal (`sarek-metal/Sarek_ir_metal.ml`)

1. Delete local `mangle_name` (line 33) and `gen_variant_def` (lines 1005–1076).
2. Update all `mangle_name` call sites to `Sarek_ir_codegen.mangle_name`.
3. **Resolve dead-code decision:**
   - Read `sarek-metal/test/test_sarek_ir_metal.ml` lines 199–222 (per KB).
   - If the test exercises a kernel with a variant type and passes today:
     the gap is intentional — do NOT add the `List.iter` call; add a comment in
     `generate_with_types`:
     ```ocaml
     (* NOTE: Metal variant type codegen not yet emitted — gen_variant_def_glsl
        equivalent lives in Sarek_ir_codegen but is not wired here. See audit
        brief backend-variant-dedup-intake.md for context. *)
     ```
   - If no variant-type test exists for Metal: add the missing call after the record
     defs block in `generate_with_types`:
     ```ocaml
     List.iter
       (Sarek_ir_codegen.gen_variant_def
          ~type_of_elttype:metal_type_of_elttype
          ~constructor_prefix:"static inline"
          buf)
       k.kern_variants ;
     ```
4. `dune build` after.

### Step 7 — Migrate Vulkan (`sarek-vulkan/Sarek_ir_glsl.ml`)

1. Delete local `mangle_name` (line 33) and `gen_variant_def` (lines 969–1036).
2. Update all `mangle_name` call sites to `Sarek_ir_codegen.mangle_name`.
   Call sites at lines: 153, 154, 269, 279, 956 (in `gen_record_def`), 970 (now gone).
3. Update call site in `generate_with_types` (line 1055, renumbered after deletion):
   ```ocaml
   List.iter
     (Sarek_ir_codegen.gen_variant_def_glsl
        ~type_of_elttype:glsl_type_of_elttype
        buf)
     k.kern_variants ;
   ```
4. `dune build` after.

### Step 8 — Full verification

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote
./scripts/check-license-headers.sh
```

All must pass. If `dune fmt` produces changes, stage them before the commit.
If `check-license-headers.sh` fails, the new files are missing SPDX headers — add them.

## Points of attention

- **Partial application + labelled args**: `List.iter (Sarek_ir_codegen.gen_variant_def ~type_of_elttype:X ~constructor_prefix:"..." buf)` produces `string * ... -> unit` — correct for `List.iter`. Do not eta-expand unless the compiler complains.
- **`Sarek_ir_codegen.mangle_name` inside `type_of_elttype`**: the recursive `*_type_of_elttype` functions call `mangle_name` internally (e.g. `| TRecord (name, _) -> mangle_name name`). These must be updated too — use the grep output from Step 4 as a checklist.
- **License headers**: `Sarek_ir_codegen.ml` and `.mli` must have SPDX headers before the license check runs.
- **Do not change generated output**: the only observable change is replacing local function definitions with calls to the shared module. If you notice any output difference in the buffer contents, stop and investigate.

## Quality gates

```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote
./scripts/check-license-headers.sh
```
