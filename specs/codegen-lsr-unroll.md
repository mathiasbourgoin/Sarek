---
name: roster-spec
type: spec
status: live
feature: Native logical shift right (lsr) on every backend; automatic unroll hint for small constant-trip for loops
brief: a downstream user's kernels-vs-ggml report, causes 2(a) and 2(b)
date: 2026-10-10
version: 1.0.0
---

# Spec — codegen-lsr-unroll

Driven by a downstream LLM inference engine's kernels-vs-ggml report (tartarus, RX 7900 XTX, 2026-10-10).
Decisions not dictated by the brief were **made autonomously by the agent**
under the owner's delegation and are marked *(agent decision)*. Owner rule:
semantics-preserving optimisations may be default; anything that can change
results must be opt-in.

## Findings this spec answers

| ID | Finding (downstream, measured) | Answer here |
|---|---|---|
| F-1 | `x lsr s` on int32 is emitted as `((s == 0) ? x : ((x >> s) ^ ((x >> 31) << ((32 - s) & 31))))`. The unrolled q5_K body has 488 VALU instructions against 236 for a hand-written one that uses plain unsigned shifts. | A logical shift becomes its own IR operator and every backend emits its native unsigned shift. |
| F-2 | Constant-trip `for` loops stay rolled; adding `_Pragma("unroll")` by hand took q5_K matvec from 40.7 to 35.9 µs. | Small constant-trip loops get an unroll hint by default on the backends that have one. |

## Clarifications

| Q | A |
|---|---|
| Why a new IR operator instead of recognising the expansion tree in each backend? *(agent decision)* | The C backends cannot infer an operand's type from an IR expression, and pattern-matching a 9-node tree in six emitters is fragile. A dedicated operator is evaluated once, so the restriction "lsr operands must be a variable or a constant" (which existed only because the expansion duplicated its operands) is lifted. The comment in `Sarek_lower_ir` that forbade new IR nodes cited the formal PTX model; that model has its own operator type (`PtxTypes.v`), and its conformance test declares its own copy of the operator type rather than mapping the production one, so nothing in `formal/` changes. |
| Operator names? *(agent decision)* | `Lshr32` and `Lshr64` in both `Sarek_ir_ppx.binop` and `Sarek_ir_types.binop`. The width is part of the operator because the C-family emitters must spell the unsigned type and cannot recover it from the operands. |
| Spelling per backend? | CUDA/HIP `((int)((unsigned int)(a) >> (b)))`, 64-bit `((long long)((unsigned long long)(a) >> (b)))`. OpenCL and Metal `((int)((uint)(a) >> (b)))`, 64-bit `((long)((ulong)(a) >> (b)))`. GLSL `int(uint(a) >> uint(b))`, 64-bit `int64_t(uint64_t(a) >> uint(b))`. WGSL `bitcast<i32>(bitcast<u32>(a) >> u32(b))`; 64-bit is refused (WGSL has no 64-bit integers, as for every other int64 construct). PTX `shr.b32` / `shr.b64`. Interpreter `Int32/Int64.shift_right_logical`. Native already used `Int32/Int64.shift_right_logical` and is unchanged. |
| Shift amounts outside `0..width-1`? | Unspecified, as in OCaml (`Int32.shift_right_logical`: "unspecified if y < 0 or y >= 32") and as before this change (the old expansion also shifted by the raw amount). Tests cover `0..31` (and `0..63` for int64) exhaustively; nothing is promised outside. |
| Unroll: default or opt-in? *(agent decision)* | Default. Unrolling a loop whose trip count is a compile-time constant does not change the order or the set of operations, so results are identical; the risk is code size and register pressure only, bounded by the threshold below. Opt-out: an explicit `pragma [...]` on the loop wins (e.g. `pragma ["unroll 1"]`), and `SAREK_OPT_DISABLE=unroll` disables the pass (read when the program starts; `Execute` memoises generated source per kernel keyed on the policy). |
| Threshold? *(agent decision)* | A loop is hinted when both bounds are integer constants (literals, or `+ - *` of literals), its trip count is between 1 and 16, and trip count × the largest expansion of the hinted loops nested in its body is at most 64. q5_K's `for wi = 0 to 7` around `for k = 0 to 3` gives 32; q6_K's `for i = 0 to 15` gives 16. A loop with an unknown or zero trip count is never hinted. |
| Which backends get a hint? | CUDA, HIP (through the CUDA generator), OpenCL and Metal: `#pragma unroll`. GLSL/Vulkan: `[[unroll]]` under `GL_EXT_control_flow_attributes`. WGSL and PTX have no source-level unroll hint: nothing is emitted. Native and the Interpreter do not compile source. |
| Where does the pass run? *(agent decision)* | At code generation (`Sarek_ir_unroll.annotate`), called by the CUDA, OpenCL, Metal and GLSL generators on the kernel body and helper bodies. The IR built by the PPX is unchanged, so IR golden tests and the interpreter are unaffected. |

## User Stories

### US-1: Logical shift is one instruction (P0)
As a kernel author, I want `x lsr s` to compile to the hardware's unsigned shift so that bit extraction costs what it costs in hand-written code.
**Independent Test**: unit codegen tests per backend; HIP ISA instruction count of a q5_K-like kernel.
1. **Given** `x lsr s` on int32, **When** CUDA source is generated, **Then** it contains `((int)((unsigned int)(x) >> (s)))` and no `^`.
2. **Given** the same kernel, **When** PTX is generated, **Then** it contains `shr.b32` and no `xor`.

### US-2: Same results as OCaml on every backend (P0)
As a kernel author, I want `lsr` to keep OCaml's semantics on every device.
**Independent Test**: e2e `test_lsr_unroll` on Native, Interpreter and every enumerated GPU device.
1. **Given** x in {0, 1, -1, min_int, max_int, -16, 0x80000001, 0x12345678, 0xdeadbeef, …} and s in 0..31, **When** `x lsr s` runs on any device, **Then** the result equals `Int32.shift_right_logical x s`.
2. **Given** an int64 x and s in 0..63, **When** run on a device with int64 support, **Then** the result equals `Int64.shift_right_logical x s`.
3. **Given** a non-trivial operand `(a.(i) + 1l) lsr s`, **When** the kernel is compiled, **Then** the PPX accepts it (it used to refuse) and the result matches OCaml.

### US-3: Small constant loops get an unroll hint (P0)
As a kernel author, I want small constant-trip loops unrolled without writing a pragma.
**Independent Test**: unit tests on `Sarek_ir_unroll` and per-backend source; e2e loop kernels give identical results.
1. **Given** `for k = 0 to 3` inside `for wi = 0 to 7`, **When** CUDA/OpenCL/Metal source is generated, **Then** both loops are preceded by `#pragma unroll`.
2. **Given** `for i = 0 to n - 1` with `n` a parameter, **Then** no hint.
3. **Given** `for i = 0 to 31` (trip 32), **Then** no hint.
4. **Given** three nested loops of trip 8, **Then** the innermost two are hinted (8, 64) and the outer is not (512).
5. **Given** `pragma ["unroll 2"] (for i = 0 to 7 ...)`, **Then** only the user's pragma is emitted.
6. **Given** `SAREK_OPT_DISABLE=unroll`, **Then** no automatic hint is emitted.
7. **Given** a hinted loop, **When** GLSL is generated, **Then** it has `[[unroll]]` and the extension line, and glslangValidator accepts it.

## Requirements

- **FR-001**: `Sarek_ast.Lsr` lowers to `EBinop (Lshr32 | Lshr64, a, b)` chosen by the result type (int64 → `Lshr64`, otherwise `Lshr32`); operands are lowered once and may be any expression.
- **FR-002**: Every backend that matches on `binop` emits a logical shift for `Lshr32`/`Lshr64` with the spellings in Clarifications; WGSL refuses `Lshr64` with a codegen error.
- **FR-003**: `Ir.Shr` stays arithmetic everywhere (`asr` unchanged).
- **FR-004**: The Interpreter evaluates `Lshr32`/`Lshr64` with `Int32/Int64.shift_right_logical`.
- **FR-005**: `Sarek_ir_unroll.annotate` wraps each qualifying `SFor` in `SPragma (["unroll"], _)` per the threshold; loops already under an explicit `SPragma` are left alone; `SAREK_OPT_DISABLE=unroll` (`Sarek_policy.restrict Auto_unroll`) makes it the identity (the replaced `SAREK_AUTO_UNROLL` was removed in M3).
- **FR-006**: CUDA, OpenCL, Metal and GLSL generators apply the pass to the kernel body and to helper bodies; GLSL turns an `["unroll"]` pragma on a loop into `[[unroll]]` and declares `GL_EXT_control_flow_attributes`.
- **FR-007**: Generated sources and results are tested per the user stories; HIP measurements before/after are recorded in the PR.

## Success criteria

- SC-1: e2e results identical to OCaml on every available device (tartarus: Native, Interpreter, HIP ×2, CUDA via ZLUDA if present, OpenCL, Vulkan; ladon: Metal).
- SC-2: HIP q5_K-like kernel: fewer VALU instructions and lower time than on `main`, measured with conditions stated.
- SC-3: full `dune runtest` passes.

## Out of scope

- Unroll hints with a count, partial unrolling, or a PTX-level unroll.
- Changing what out-of-range shift amounts do.
- The formal PTX model (`formal/codegen-ptx`), which has its own operator set.
