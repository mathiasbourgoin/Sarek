---
name: roster-spec
type: spec
status: live
feature: Stdlib float32 <-> int32 bit reinterpretation (Int32.float_of_bits / Int32.bits_of_float) on every backend
brief: briefs/stdlib-bitcasts-intake.md
date: 2026-10-10
version: 1.0.0
---

# Spec — Int32.float_of_bits / Int32.bits_of_float

Driver: a downstream LLM inference engine stores f32 scales inside int32 expert records copied to the
device as one block, and decodes them with a downstream intrinsic
`float_of_bits` that only the C-family backends (which read `Sarek_registry`)
can serve. This moves the operation, and its inverse, into the stdlib.

## Clarifications

| Q | A |
|---|---|
| Names? | `Int32.float_of_bits : int32 -> float32`, `Int32.bits_of_float : float32 -> int32` — OCaml's `Stdlib.Int32` names and semantics. Native maps the kernel path `Int32` to OCaml's `Int32` already. Decided autonomously by the agent (intake). |
| Which frameworks? | CUDA source (also HIP: `Hip_plugin` generates through `Sarek_ir_cuda`, framework tag "CUDA"), OpenCL, Metal (FFI-registry template); GLSL/Vulkan and WGSL (`pre_hook`); PTX (native handler); Native; Interpreter. |
| Which paths are "the stdlib Int32"? | `["Int32"]`, `["Sarek_stdlib"; "Int32"]`, `["Sarek_stdlib_meta"; "Int32"]` — the paths the existing stdlib surface already uses for Float32. |
| Pure registry entry? | None. Its values are callee names for `emit_call`; templates containing `%s` are deliberately absent (test_intrinsic_surface `is_plain_call`). |
| What is compared bit-exactly? | int32 outputs (roundtrip, encode). A float32 output read by the host is widened to double, so decode is compared through `Int32.bits_of_float` of the host value, NaN rows by NaN-ness only. |
| NaN semantics? | Devices move bits through a register. Native/Interpreter carry float32 as an OCaml double: quiet-NaN payloads survive, a signalling NaN is quieted (bit 22) — OCaml's own behaviour. |

## User Stories

### US-1: Decode an f32 stored as int32 (Priority: P0)
As a kernel author, I want `Int32.float_of_bits a.(i)` so that f32 values packed in int32 records decode on any backend.
**Why this priority**: the downstream use case.
**Scope**: does not cover f64/int64.
**Independent Test**: e2e `decode` and `record` kernels on every enumerated device.
**Acceptance Scenarios**:
1. **Given** a.(i) = 0x3f800000, **When** decode runs, **Then** b.(i) = 1.0 on every device.
2. **Given** a record [bits(2.25); 3], **When** record runs, **Then** out = 6.75.
3. **Given** a.(i) = 0x80000000, **When** decode runs, **Then** the host reads -0.0 (sign bit set).

### US-2: Encode and round trip (Priority: P0)
As a kernel author, I want `Int32.bits_of_float` so that I can store a float32 as its bit pattern.
**Scope**: no rounding mode control.
**Independent Test**: e2e `encode`, `roundtrip`.
**Acceptance Scenarios**:
1. **Given** b.(i) = -1.0, **When** encode runs, **Then** c.(i) = 0xbf800000.
2. **Given** any non-NaN pattern x, **When** roundtrip runs, **Then** c.(i) = x on every device.
3. **Given** x = 0x7f800001 (sNaN), **When** roundtrip runs, **Then** devices return x and Native/Interpreter return 0x7fc00001 (OCaml).

### US-3: Special values (Priority: P1)
As a kernel author, I want ±0, ±inf, denormals, max finite and NaN to survive so that packed data is not corrupted.
**Scope**: arithmetic on those values (FTZ, fast-math) is not covered — only the bitcast path.
**Independent Test**: the special-pattern rows of the e2e test.
**Acceptance Scenarios**:
1. **Given** 0x00000001, **When** roundtrip runs, **Then** 0x00000001 comes back on every tested device.
2. **Given** 0x7fc12345, **When** roundtrip runs, **Then** 0x7fc12345 comes back on every tested device.
3. **Given** 0x7f800000, **When** decode runs, **Then** the host reads +inf.

### US-4: Misuse is refused, other modules are not captured (Priority: P1)
As a library author with my own `float_of_bits`, I want the stdlib lowering not to capture it, and wrong arity to fail loudly.
**Independent Test**: unit test on generated sources + downstream test.
**Acceptance Scenarios**:
1. **Given** an IR call `Int32.float_of_bits` with 2 args, **When** GLSL/WGSL/CUDA/OpenCL/Metal/PTX generate, **Then** each raises (no source).
2. **Given** `Dstest_intrinsics.Bits.float_of_bits`, **When** PTX generates, **Then** it is refused as before (unsupported), and GLSL/WGSL raise unknown_intrinsic.
3. **Given** the downstream test, **When** run, **Then** CUDA/OpenCL still emit its own template.

### US-5: Spellings pinned and validated (Priority: P1)
As a maintainer, I want the per-backend spellings pinned and compiled by the gates the repo already has.
**Independent Test**: unit test asserting each spelling; GLSL through glslang, WGSL through naga, PTX through the ptxas sweep, OpenCL through clang, Metal through the Metal compile gate (where the tool exists).
**Acceptance Scenarios**:
1. **Given** a kernel using both names, **When** each backend generates, **Then** the source contains the backend's spelling and no OCaml path.
2. **Given** the PTX sweep, **When** it runs, **Then** both names have recipes that assemble.
3. **Given** Metal codegen, **When** generated, **Then** it emits `as_type<float>`, never `as_float`.

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-1 | Metal shares OpenCL's branch in `cuda_or_opencl`; the helper is duplicated in stdlib and stdlib_meta. | Three-way `bitcast_dev` in both files; unit test asserts the templates of both registrations for CUDA/OpenCL/Metal and the refusal for GLSL/WGSL. |
| C-2 | US-1 | HIP framework tag / support. | HIP generates through `Sarek_ir_cuda` ("CUDA"); hipRTC declares `__int_as_float`; measured by the e2e run on the RX 7900 XTX. |
| C-3 | US-1 | Pure registry is a third registration point. | Not registered there by design (templates with `%s`). |
| C-4 | US-1 | Host read of float32 widens. | Bit-exact checks on int32 outputs only (see Clarifications). |
| C-5 | US-2 | Native float32 may not be rounded per op. | `bits_of_float` narrows the carried double round-to-nearest (OCaml); the e2e arithmetic uses exactly representable values. Documented. |
| C-6 | US-3 | GLSL/WGSL leave NaN/inf/denormal results of a bitcast unspecified. | Measured on the devices here; any failing backend is documented as excused in the stdlib doc and the e2e expectation, never silently. WGSL literal non-finite bitcasts may be refused at shader creation (const-eval): documented. |
| C-7 | US-3 | Fast-math / FTZ. | Only the bitcast path is specified; arithmetic on denormals is out of scope. |
| C-8 | US-3 | "NaN payloads come out exactly" was unmeasured. | Claim restated as measured per device after the e2e run. |
| C-9 | US-4 | PTX dispatch ignores the path. | Handler declines (`None` -> unsupported) off the stdlib Int32 path; unit-tested. |
| C-10 | US-4 | Which paths, unqualified calls. | Paths listed in Clarifications; the PPX resolves `let open Int32` to the registered path (checked in the unit test on a PPX kernel). |
| C-11 | US-4 | Wrong operand type. | PPX typer rejects; PTX checks the register class as a backstop. |
| C-12 | US-4 | Arity guarded in several places. | Each lowering checks: emit_unary (GLSL/WGSL), placeholder count (C-family), intr_unary_arg (PTX); unit-tested on all. Not added to `intrinsic_arity`, which would also constrain other modules' same-named intrinsics. |
| C-13 | US-5 | Gates outside the list. | PTX sweep recipes added; native witness covers Float32/Float64 only (Int32 maps to OCaml's Int32, compile-checked by the e2e test). |
| C-14 | US-5 | Tool availability. | Gates skip where the tool is absent (existing behaviour); QA records which ran. |
| C-15 | US-4 | Downstream `Bits.float_of_bits` shadowing. | Downstream test must still pass unchanged. |

## Functional Requirements

#### Declaration
- **FR-001** [US-1, US-2]: The stdlib MUST declare `Int32.float_of_bits : int32 -> float32` and `Int32.bits_of_float : float32 -> int32` in both `Sarek_stdlib` and `Sarek_stdlib_meta`.
- **FR-002** [US-1]: The FFI registry MUST spell them `__int_as_float`/`__float_as_int` for CUDA (and so HIP), `as_float`/`as_int` for OpenCL, `as_type<float>`/`as_type<int>` for Metal, and MUST refuse GLSL/WGSL.

#### Lowering
- **FR-003** [US-1]: GLSL MUST emit `intBitsToFloat`/`floatBitsToInt`; WGSL `bitcast<f32>`/`bitcast<i32>`; PTX one `mov.b32` between the s32 and f32 register files.
- **FR-004** [US-1, US-2]: Native and Interpreter MUST compute OCaml's `Int32.float_of_bits`/`bits_of_float`.
- **FR-005** [US-2, US-3]: On every tested GPU device the in-kernel round trip MUST be the identity for every tested pattern, including sNaN and NaN payloads, or the deviation MUST be documented.

#### Refusal
- **FR-006** [US-4]: GLSL, WGSL and PTX MUST NOT lower a same-named intrinsic from another module path with the stdlib lowering.
- **FR-007** [US-4]: Every backend MUST raise on a call with an argument count other than 1.

## Acceptance Criteria

- AC-1 [US-1]: decode + record kernels on Native, Interpreter, HIP, OpenCL, Vulkan, PTX (ZLUDA, if it enumerates), Metal (ladon) → all rows PASS.
- AC-2 [US-2, US-3]: encode + roundtrip on the same devices → all rows PASS with the expectations in US-2.
- AC-3 [US-5, C-1]: unit test: templates for both registrations and the five source backends' spellings → pass; Metal never `as_float`.
- AC-4 [US-4, C-9, C-12]: unit test: off-path call refused on GLSL/WGSL/PTX; arity 2 refused on all six codegens → pass.
- AC-5 [US-5]: PTX sweep, golden/naga/glslang, downstream test → pass.

## Edge Cases

- EC-1 [US-1]: 0x80000000 → -0.0 with sign.
- EC-2 [US-3]: 0x00000001, 0x007fffff, 0x807fffff → exact on devices that ran (measured).
- EC-3 [US-3]: 0x7f800001 sNaN → devices: identity (measured); Native/Interpreter: 0x7fc00001.
- EC-4 [US-4]: `Bits.float_of_bits` on PTX → unsupported, as before.
- EC-5 [US-3]: WGSL bitcast of a literal non-finite pattern → may be refused by shader creation (const-eval); documented, not tested at runtime (no WebGPU device).

## Runnable Checks

- CHECK-1 [AC-1, AC-2]: `dune exec sarek/tests/e2e/test_int32_bitcasts.exe` (tartarus; with `LD_LIBRARY_PATH=$HOME/opt/zluda` for PTX; on ladon for Metal) → exit 0, every device PASS.
- CHECK-2 [AC-3, AC-4]: `dune exec sarek/tests/unit/test_int32_bitcasts_codegen.exe` → exit 0.
- CHECK-3 [AC-5]: `dune build @sarek/tests/unit/runtest @sarek/tests/codegen_golden/runtest @sarek/tests/downstream/runtest` → exit 0.

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"stdlib-bitcasts","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-001","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-002","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-003","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-004","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-005","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-006","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-007","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-003","FR-004"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-004","FR-005"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-001","FR-002","FR-003"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-006","FR-007"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-003","FR-006"]}
{"record":"check","id":"CHECK-1","for":["AC-1","AC-2"]}
{"record":"check","id":"CHECK-2","for":["AC-3","AC-4"]}
{"record":"check","id":"CHECK-3","for":["AC-5"]}
```

## Entities

- `bit reinterpretation`: reuse of the 32 bits of one scalar as another type, with no numeric conversion.
