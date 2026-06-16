---
name: roster-spec
type: spec
status: live
feature: PTX DShared emission with formal verification
brief: briefs/ptx-dshared-formal-intake.md
date: 2026-06-16
version: 1.0.0
---

# Spec — PTX DShared Emission with Formal Verification

## Clarifications

| Q | A |
|---|---|
| Should `env` type change or use a parallel structure for shared tracking? | Separate `arr_memspaces : (string, unit) Hashtbl.t` field on `reg_alloc` — avoids changing the env type signature or downstream lookup sites |
| Are shared memory pointers `.u32` or `.u64`? | `.u32` — PTX shared state space uses 32-bit addresses; `ld.shared` uses the 32-bit register directly, no `cvta` needed |
| What happens when `DShared` has `size_opt = None`? | Raise `Ptx_codegen_error "DShared: dynamic shared memory (size=None) not yet supported"` — out of scope for this cycle |
| What Rocq record shape for `ir_shared_decl`? | `{ sd_name : string; sd_elt : ir_elt_type; sd_size : nat }` — alignment derived from element type, not stored explicitly |
| Should `DShared` populate `alloc.arr_elt_types`? | Yes — so `infer_elt_type` finds the shared array and chooses the right `ld.shared.*` variant |
| Is `.b32` or `.f32` used in `.shared` declarations? | `.bNN` (bit-sized) in declarations, typed qualifier (`f32`, `s32`, `f64`, `s64`) in `ld.shared`/`st.shared` instructions |

## User Stories

### US-1: PTX DShared Code Generation (Priority: P0)

As a Sarek kernel author, I want a kernel with `DShared` declarations to compile to correct PTX with `.shared` directives and `ld.shared`/`st.shared` instructions, so that shared memory kernels run on NVIDIA GPUs without NVRTC.

**Why this priority**: Without this, every shared-memory kernel (reductions, prefix sums, tiled matmul) hits `unsupported` and falls back to CUDA C. It is the highest-impact single gap in the PTX emitter.

**Scope**: This story does NOT cover dynamic shared memory (`size_opt = None`), `DParam` with `arr_memspace = Shared`, `EArrayCreate` in shared space, or multiple `DShared` declarations with the same name.

**Independent Test**: Compile a `vec_add_shared` kernel (uses a 256-element TFloat32 DShared buffer) with `Sarek_ir_ptx_kernel.generate` and assert the output string contains `.shared .align 4 .b32 shmem[256]`, `mov.u32`, and `ld.shared.f32`.

**Acceptance Scenarios**:

1. **Given** a kernel with `DShared("shmem", TFloat32, Some (EConst (I32 256)))` in `kern_locals` and a body that reads `EArrayRead("shmem", idx)`, **When** `Sarek_ir_ptx_kernel.generate` is called, **Then** the PTX output contains `.shared .align 4 .b32 shmem[256];`, `mov.u32 %r<N>, shmem;`, and `ld.shared.f32 %f<M>, [%r<N>+<offset>];`.

2. **Given** a kernel with `DShared("tmp", TInt32, Some (EConst (I32 64)))` and a body that writes to `tmp`, **When** PTX is generated, **Then** the output contains `st.shared.s32` instructions, not `st.global`.

3. **Given** a kernel with `DShared("x", TFloat32, None)` (dynamic size), **When** `Sarek_ir_ptx_kernel.generate` is called, **Then** `Ptx_codegen_error` is raised with a message containing "dynamic shared memory".

4. **Given** a kernel with `DShared("x", TFloat32, Some (EConst (I32 0)))` (zero size), **When** PTX generation is attempted, **Then** `Ptx_codegen_error` is raised with a message containing "size must be positive".

5. **Given** a kernel that has both a global `DParam("v", TFloat32)` and `DShared("shmem", TFloat32, Some (EConst (I32 32)))`, **When** PTX is generated, **Then** reads from `v` use `ld.global.f32` and reads from `shmem` use `ld.shared.f32`.

6. **Given** the codegen-ptx conformance suite in `test/test_codegen_ptx_conformance.ml`, **When** `dune runtest formal/codegen-ptx/test/` is run, **Then** all tests pass including at least 2 new tests covering scenarios AC-1 and AC-2 above.

---

### US-2: Formal Proof of DShared Kernel Correctness (Priority: P0)

As a formal verification engineer, I want the `emit_kernel_correct` theorem in `PtxKernelSpec.v` to cover kernels with `kern_shared` declarations, so that DShared emission is certified correct under the agpu semantics with 0 admits.

**Why this priority**: The 0-admits invariant is absolute. If the Rocq spec is not extended to cover the new `kern_shared` field before merging, the formal apparatus no longer reflects the OCaml implementation — the proof covers a strict subset of the code.

**Scope**: This story does NOT cover proving correctness of the `ld.shared` PTX instruction itself (modelled as direct `shared_mem` reads in `AGpuSemantics`), loops over shared arrays, or dynamic shared memory allocation.

**Independent Test**: Run `coqc theories/PtxKernelSpec.v` in `formal/codegen-ptx/` and assert exit code 0; then run `coqchk -R . CodegenPtx PtxKernelSpec` and assert output contains "Modules were successfully checked". Assert `grep -cE '^\s*(Admitted\.|admit\.)' theories/PtxKernelSpec.v` returns 0.

**Acceptance Scenarios**:

1. **Given** `ir_kernel` is extended with `kern_shared : list ir_shared_decl` where `ir_shared_decl` has fields `sd_name : string`, `sd_elt : ir_elt_type`, `sd_size : nat`, **When** `coqc theories/PtxKernelSpec.v` is run, **Then** it exits 0 with 0 `Admitted` or `admit` in the file.

2. **Given** `emit_kernel_correct` is stated as `forall k st, agpu_exec_ir_kernel st k = agpu_exec_ptx_kernel st (emit_ast_kernel k)` where both sides ignore `k.(kern_shared)`, **When** `coqchk` is run on the compiled `.vo`, **Then** the output contains "Modules were successfully checked".

3. **Given** an `ir_kernel` with `kern_shared = [{ sd_name := "shmem"; sd_elt := TInt32; sd_size := 256 }]` but a body that performs no shared memory access, **When** the `emit_kernel_correct` theorem is instantiated for this kernel, **Then** it holds without additional lemmas (the non-interference property: shared decls do not affect body execution).

4. **Given** `formal/codegen-ptx/theories/PtxKernelSpec.v` after the extension, **When** `grep -cE '^\s*(Admitted\.|admit\.)' theories/PtxKernelSpec.v` is run, **Then** the output is `0`.

---

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-1 | Which PTX idiom loads a shared symbol: `mov.u32` or `cvta.to.shared`? | PTX 8.0 target: `mov.u32 %r, shmem;` is valid for `.shared` symbol labels. `cvta` is needed only when converting from generic address space. Not used here. |
| C-2 | US-1 | `.b32` vs `.f32` in `.shared` declaration: semantically correct? | PTX uses bit-sized types in `.shared` declarations; typed qualifiers go on `ld.shared.*`/`st.shared.*`. `.b32` + `ld.shared.f32` is correct PTX idiom. |
| C-3 | US-1 | DShared name collides with DParam name — undefined behavior? | User error; out of scope this cycle. `emit_locals` runs after `emit_params`, so the DShared binding overwrites the DParam binding in env. Documented as undefined behavior. |
| C-4 | US-1 | Does `Ptx_codegen_error` exist? | Yes — `Sarek_ir_ptx_types.ml:13`: `exception Ptx_codegen_error of string`. Already used by all `unsupported` calls. |
| C-5 | US-2 | Does adding `kern_shared` break existing proofs? | No — only `PtxKernelSpec.v` uses `ir_kernel`. `agpu_exec_ir_kernel` only accesses `k.(kern_body)`; proof reduces to `emit_stmt_correct` unchanged. |
| C-6 | US-2 | Does coqchk only verify a concrete instance, not the forall? | No — `coqchk` verifies the entire compiled `.vo` including the `forall k st` quantified theorem. |
| C-7 | US-2 | AC-3 is trivially true — does it prove anything? | It proves the non-interference property (adding kern_shared does not affect body correctness), which IS the key claim. It is the expected proof shape. |

## Functional Requirements

### OCaml PTX Emitter — DShared Support

- **FR-001** [US-1]: `emit_locals` in `Sarek_ir_ptx_kernel.ml` MUST emit a `.shared .align N .bXX name[size];` directive for each `DShared(name, elt, Some (EConst (I32 n)))` where `n > 0`.
- **FR-002** [US-1]: After emitting the `.shared` directive, the emitter MUST emit `mov.u32 %r<k>, name;` and bind `name → %r<k>` in `env` and mark `name` in `alloc.arr_memspaces`.
- **FR-003** [US-1]: `emit_array_read` in `Sarek_ir_ptx_mem.ml` MUST accept a `~is_shared:bool` parameter and emit `ld.shared.*` when `true`, `ld.global.*` when `false`.
- **FR-004** [US-1]: `emit_array_write` in `Sarek_ir_ptx_mem.ml` MUST accept a `~is_shared:bool` parameter and emit `st.shared.*` when `true`, `st.global.*` when `false`.
- **FR-005** [US-1]: All call sites of `emit_array_read`/`emit_array_write` in `Sarek_ir_ptx_expr.ml` MUST pass `~is_shared:(Hashtbl.mem alloc.arr_memspaces arr_name)`.
- **FR-006** [US-1]: `emit_locals` MUST raise `Ptx_codegen_error` containing `"dynamic shared memory"` when `size_opt = None`.
- **FR-007** [US-1]: `emit_locals` MUST raise `Ptx_codegen_error` containing `"size must be positive"` when `size_opt = Some (EConst (I32 n))` and `n <= 0`.
- **FR-008** [US-1]: `DShared(name, elt, _)` MUST register `name` in `alloc.arr_elt_types` so `infer_elt_type` finds it.
- **FR-009** [US-1]: At least 2 new Alcotest conformance tests for DShared MUST be added to `test/test_codegen_ptx_conformance.ml`.

### Rocq Formal Proof — DShared Extension

- **FR-010** [US-2]: `PtxKernelSpec.v` MUST define `ir_shared_decl` as a record with fields `sd_name : string`, `sd_elt : ir_elt_type`, `sd_size : nat`.
- **FR-011** [US-2]: `ir_kernel` MUST be extended with field `kern_shared : list ir_shared_decl`.
- **FR-012** [US-2]: `ptx_kernel_ast` MUST be extended with field `ptx_kern_shared : list ptx_shared_decl` where `ptx_shared_decl` mirrors `ir_shared_decl`.
- **FR-013** [US-2]: `agpu_exec_ir_kernel` MUST NOT access `kern_shared` (the field is semantically neutral).
- **FR-014** [US-2]: `agpu_exec_ptx_kernel` MUST NOT access `ptx_kern_shared` (static directive, semantically neutral).
- **FR-015** [US-2]: `emit_ast_kernel` MUST map `kern_shared` to `ptx_kern_shared` element-wise.
- **FR-016** [US-2]: `emit_kernel_correct` MUST be re-proved with 0 `Admitted` or `admit` in the file.
- **FR-017** [US-2]: An `Example` witness kernel with non-empty `kern_shared` MUST be added to `PtxKernelSpec.v` per Rule 11 (non-vacuousness gate).

## Acceptance Criteria

- AC-1 [US-1]: DShared TFloat32 static emit → output contains `.shared .align 4 .b32 shmem[256];` and `ld.shared.f32`
- AC-2 [US-1]: DShared TInt32 static emit → output contains `st.shared.s32`
- AC-3 [US-1]: DShared size=None → `Ptx_codegen_error` raised with "dynamic shared memory"
- AC-4 [US-1]: DShared size=0 → `Ptx_codegen_error` raised with "size must be positive"
- AC-5 [US-1]: Mixed global+shared kernel → global uses `ld.global`, shared uses `ld.shared`
- AC-6 [US-1]: `dune runtest formal/codegen-ptx/test/` passes with ≥2 new DShared tests
- AC-7 [US-2]: `ir_kernel` extended with `kern_shared` field, `coqc PtxKernelSpec.v` exits 0
- AC-8 [US-2]: `coqchk` reports "Modules were successfully checked"
- AC-9 [US-2]: `grep -cE '^\s*(Admitted\.|admit\.)' formal/codegen-ptx/theories/PtxKernelSpec.v` returns 0
- AC-10 [US-2]: `Example` witness with non-empty `kern_shared` present in `PtxKernelSpec.v`

## Edge Cases

- EC-1 [US-1]: Two `DShared` declarations with the same name → last binding wins in env; undefined behavior, document only, no runtime check
- EC-2 [US-1]: `DShared` with size=0 → `Ptx_codegen_error "size must be positive"` (covered by AC-4)
- EC-3 [US-1]: Kernel with zero `DParam` and non-empty `DShared` → empty `.param` section emitted (valid PTX, already handled by `emit_params` returning `""`)
- EC-4 [US-1]: `EArrayRead` of a shared array with a type mismatch between DShared elt and read type → `infer_elt_type` uses the DShared-registered type, so the declared type wins; no cross-check in this cycle
- EC-5 [US-2]: `kern_shared` list is empty → proof is identical to pre-extension case; must hold trivially

## Runnable Checks

- CHECK-1 [AC-1]: `dune exec -- ocaml -e 'let _ = Sarek_ir_ptx_kernel.generate kernel_with_dshared_float32 in ()' 2>&1 | grep -c "ld.shared.f32"` → expected: `1`
- CHECK-2 [AC-3]: Alcotest test `"DShared dynamic raises"` in `test_codegen_ptx_conformance.ml` → expected: test passes, exception caught
- CHECK-3 [AC-6]: `dune runtest formal/codegen-ptx/test/` → expected: exit 0, `N tests` with ≥2 new DShared tests
- CHECK-4 [AC-7]: `cd formal/codegen-ptx && make -f CoqMakefile 2>&1 | tail -5` → expected: no errors, exit 0
- CHECK-5 [AC-8]: `cd formal/codegen-ptx && coqchk -R . CodegenPtx PtxKernelSpec 2>&1 | grep "successfully"` → expected: "Modules were successfully checked"
- CHECK-6 [AC-9]: `grep -cE '^\s*(Admitted\.|admit\.)' formal/codegen-ptx/theories/PtxKernelSpec.v` → expected: `0`

## Entities

- `DShared`: An IR declaration variant `DShared of string * elttype * expr option` representing a named shared memory buffer in a kernel
- `ir_shared_decl`: Rocq record `{ sd_name : string; sd_elt : ir_elt_type; sd_size : nat }` — formal model of a static DShared declaration
- `arr_memspaces`: Runtime `(string, unit) Hashtbl.t` field on `reg_alloc` tracking which array names are in shared memory space, used to choose `ld.shared` vs `ld.global`
- `emit_kernel_correct`: Top-level Rocq theorem stating IR kernel execution equals PTX kernel execution under agpu semantics, for all kernels and initial states
