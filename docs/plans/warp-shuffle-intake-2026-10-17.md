# Intake: `warp_shuffle`, `warp_shuffle_down`, `warp_shuffle_up`, `warp_shuffle_xor` on every backend

**Status: VALIDATED** — owner wrote « Validé » in the chat on 2026-10-17, right after receiving the
summary of this brief and the decisions D1-D7 recorded in section 13 (the validation targets this brief
with these decisions).

Branch `feat/warp-shuffle` (worktree `/mnt/ssd-external-2to/wt-warp-shuffle`, rebased on
`origin/main` 82fdd6b8, which includes the launch bounds PR 520). Validated 2026-10-17;
decisions in section 13. Task type: **api-change**.

## 1. Problem

The four shuffle primitives are declared (`sarek/ppx/Sarek_core_primitives.ml:662-695`,
category `"warp"`, `ThreadVarying`, `WarpConvergence`, `Pure`) and checked by the
convergence rule, but **no backend lowers them**: the PPX refuses them at the call site
as "not lowered by any backend yet" (`sarek/ppx/Sarek_unlowered.ml:38-41`,
`Sarek_error.Warp_collective`), and every IR backend raises `Unknown_intrinsic`
(`sarek/tests/codegen_golden/test_intrinsic_fallback_all.ml`, which uses `warp_shuffle` as
its probe). `docs/optimization/opt-expressivity-gaps.md` gap 1 (P0) calls this the
highest-leverage item and the prerequisite of gaps 5 and 13 (block reduce/scan, stdlib).

Downstream motivation (a downstream LLM inference engine): the reduction of `gdn_seq` (8 lanes per column) goes through shared
memory with 2 barriers per token. llama.cpp reduces by shuffles
(`ggml-cuda/gated_delta_net.cu:5-100`). Without shuffles the engine cannot remove those
barriers (perf claim to be measured, not asserted: see section 10, AC-9).

## 2. Owner constraint (binding)

The Sarek compiler must handle warp shuffle **on all backends, Native included**, if
necessary by a slow sequential version. A backend lacking the hardware feature must
either (a) emulate (Native, Interpreter: mandatory; others: see D5) or (b) refuse with
a located, named error at generation or launch time. Silent wrong results and late
driver errors are not acceptable.

## 3. Goal

Make these four primitives callable from `[%kernel]` code with ONE exactly specified
semantics (section 4), lowered on CUDA C (NVRTC), PTX (direct), HIP, OpenCL, Vulkan
(GLSL), Metal, WGSL, Native and Interpreter (the jsoo interpreter inherits the
Interpreter), keep the divergence refusal, and ship tests, coverage table and docs.

## 4. Semantics (the contract every backend implements)

Notation: `W` = warp width of the device (32 or 64 for CUDA/HIP/AMD; 32 on Apple GPUs;
8..64 for Vulkan/OpenCL subgroups); `lane` = the calling thread's index in its warp;
`v` = the value, `x` the lane/delta/mask operand (int32).

1. **Warp and lane.** A warp is the hardware subgroup. For a block of linear size
   `n = bx*by*bz`, linear thread id `t = tx + bx*(ty + by*tz)` (x fastest) and
   `lane = t mod W`, warp index `t / W`. Backends where the hardware lane is not
   architecturally tied to `t` (Vulkan, OpenCL) must prove the equality with a
   device self-test (section 6.5) or refuse.
2. **Participation (mask).** All `W` lanes of the warp participate; the mask is always
   the full warp mask and is not user-visible. This is sound only because the
   convergence rule (section 7) rejects calls in diverged flow. A launch whose kernel
   uses a shuffle and whose block size is not a multiple of `W` is **refused at launch**
   (`Invalid_launch`, naming kernel, block size, `W`), because lanes of a partial warp
   would read non-existent sources (undefined in CUDA/HIP/Metal).
3. **Result per primitive (lane `i`, source `s`):**

   | Primitive | Source lane `s` | Out of range |
   |---|---|---|
   | `warp_shuffle v x` | `x land (W-1)` (`x` read as two's complement; `W` is a power of two) | cannot occur |
   | `warp_shuffle_down v d` | `i + d` | if `d < 0` or `i + d >= W`: own `v` |
   | `warp_shuffle_up v d` | `i - d` | if `d < 0` or `i - d < 0`: own `v` |
   | `warp_shuffle_xor v m` | `i lxor m` | if `m < 0` or `i lxor m >= W`: own `v` |

   This is CUDA's `__shfl_*_sync(full_mask, v, x)` with `width = W`, **specified
   independently of any vendor's undefined behaviour**: where a native instruction is
   undefined or uses only the low 5 bits of the operand (to be confirmed against the
   PTX ISA; hence the guards and the boundary tests below), the emitter adds an explicit
   guard so the table above holds exactly. Boundary values to test: `x` in
   `{0, 1, W-1, W, W+1, -1, 2*W+1}`.
4. **Early exit.** A lane that already returned is an inactive source. The
   convergence rule's F-04 (varying early return makes the rest `Diverged`) already
   refuses the shuffle after it; the brief keeps that and adds a negative test.
5. **Types.** The primitives are `int32 -> int32 -> int32` today. The engine needs
   `float32`. Required: `int32`, `float32`; `float64` where the backend has it
   (64-bit shuffles as two 32-bit halves; Metal has no float64: refuse with the
   existing Metal float64 error). `bool`, vectors and records are out of scope.
   Typing mechanism: decision D1.
6. **Sub-warp segments.** No `width` parameter in this task (decision D2). `xor` with
   masks below a power of two `P <= W` stays inside aligned `P`-lane groups on every
   backend, which is what the engine's 8-lane reduction needs. `down`/`up` inside a segment
   needs an explicit `width`; out of scope, documented.
7. **Purity and motion.** A shuffle is *convergent*: its result depends on other
   lanes, so it must not be hoisted, sunk, duplicated, or moved across control flow,
   and must not be treated as `Pure` by any IR pass (`purity = Pure` in the table is
   wrong for motion purposes). See section 8, risk R2.

## 5. Front end and IR work

- `Sarek_core_primitives.ml`: typing per D1; purity/motion flag per section 4.7; a
  property of the table "lowered" so that `Sarek_unlowered.ml` refuses only the
  still-unlowered warp primitives (`warp_vote_*`, `warp_ballot`, `warp_active_mask`,
  `warp_size` stay refused with the same messages and tests).
- `Sarek_stdlib` (`Gpu`/`Std` paths): the four names resolvable as today's `warp_ballot`
  paths are (`Gpu.warp_shuffle`, `Std.warp_shuffle`, bare).
- `Sarek_ir_intrinsic_dispatch.ml`: arity table entries (2 args each) so arity errors
  are central; backend arms in section 6.
- A kernel-level flag `uses_warp_collective` (computed like `expr_uses_barriers`,
  `Sarek_convergence.ml:355`) carried to the launcher: forces BSP/barrier mode in
  Native and Interpreter, drives the launch check (4.2) and the passes' refusal (R2).
- Update the comments that say "when warp_shuffle is lowered, assert message 2 again"
  (`sarek/tests/negative/test_warp_diverged.ml`, `docs/optimization/opt-expressivity-gaps.md`
  line ~82, `specs/convergence-semantics.md` §7 row, `sarek/sarek/BSP.md` "Warp-Level
  Operations", `Sarek_error.ml:219-228` text).

## 6. Lowering per backend

All emit the section-4 table; "guard" = explicit compare so that out-of-range yields
own `v`. `W` and `lane` come from the backend's own built-ins, never a constant 32.

| Backend | Lowering | Capability test / refusal |
|---|---|---|
| CUDA C (NVRTC), `Sarek_ir_cuda.ml` | `__shfl_sync(0xffffffffu, v, x & 31)`, `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` (`width=32`, so `W=32`), float/double native, guards for `d`/`m` outside `[0,32)` | sm >= 3.0 always true for supported devices; ZLUDA: run the e2e and record result, no assumption |
| PTX direct, `Sarek_ir_ptx_expr.ml` (`emit_intrinsic_native`, owned separately: coordinate) | `shfl.sync.{idx,down,up,bfly}.b32 d\|p, a, b, c, 0xffffffff;` (`c` = 0x1f, up uses 0), float via `mov.b32`, f64 via `mov.b64 {lo,hi}` and two shuffles | needs PTX ISA >= 6.0 / sm_30 + `.version`; the existing ISA gate decides, refusal names the minimum |
| HIP, `Sarek_ir_cuda.ml` HIP dialect | `__shfl(v, x)`, `__shfl_down`, `__shfl_up`, `__shfl_xor` with `width = warpSize` (device built-in, 32 on RDNA, 64 on CDNA/GCN), guards use `warpSize`; verify whether the installed ROCm needs `__shfl_*_sync` with a 64-bit mask | none beyond the framework `warp_size`; test gfx1100 (W=32) here, W=64 only if hardware available, otherwise unit test on a faked W=64 in Native |
| OpenCL, `Sarek_ir_opencl.ml` | `sub_group_shuffle`, `sub_group_shuffle_xor` (`cl_khr_subgroup_shuffle`), `sub_group_shuffle_up/down` (`cl_khr_subgroup_shuffle_relative`), lane/width from `get_sub_group_local_id()` / `get_sub_group_size()`, `#pragma OPENCL EXTENSION ... : enable` emitted only when used; `cl_intel_subgroups` shuffle as secondary spelling if measured | Device discovery reads `CL_DEVICE_EXTENSIONS` and stores the two flags in the capabilities; absent: located `Unsupported_feature {backend="OpenCL"; feature="warp_shuffle"; requires="cl_khr_subgroup_shuffle[_relative]"}` at generation (or emulation, D5). **Fix needed**: `Opencl_plugin_base.ml:127` hard-codes `warp_size = 32`; it must be the real sub-group size or the shuffle guards are wrong |
| Vulkan GLSL, `Sarek_ir_glsl.ml` | `#extension GL_KHR_shader_subgroup_shuffle : require` (+ `_shuffle_relative` for up/down); `subgroupShuffle`, `subgroupShuffleXor`, `subgroupShuffleUp/Down`; `W = gl_SubgroupSize`, `lane = gl_SubgroupInvocationID`; guards on `gl_SubgroupSize`; float32 native, float64 under the existing fp64 extension; SPIR-V target env must be Vulkan 1.1 or higher (check the current glslang target) | `VkPhysicalDeviceSubgroupProperties.supportedOperations` must contain `SHUFFLE` (0x10) and, for up/down, `SHUFFLE_RELATIVE` (0x20), and `supportedStages` must contain compute (the struct is already read at `Vulkan_api_device.ml:506-547`, extend it); absent: located refusal. Pipelines needing full subgroups: `REQUIRE_FULL_SUBGROUPS` or the launch check 4.2 |
| Metal, `Sarek_ir_metal.ml` | `simd_shuffle`, `simd_shuffle_down`, `simd_shuffle_up`, `simd_shuffle_xor`, with `[[thread_index_in_simdgroup]]` and `[[threads_per_simdgroup]]` added to the kernel parameters only when a shuffle is used; guards as table (Metal leaves out-of-range undefined) | Apple GPUs family >= 4 / macOS 10.14+ (SIMD-group functions); no float64 on Metal: existing refusal |
| WGSL, `Sarek_ir_wgsl.ml` | `enable subgroups;` + `subgroupShuffle`, `subgroupShuffleXor`, `subgroupShuffleUp/Down`, `subgroup_invocation_id`, `subgroup_size` built-ins | WebGPU "subgroups" feature required; no runtime device exists in CI: golden text only, absent feature = located refusal; runtime marked *unverified* in the coverage table |
| Native (PPX-generated OCaml) | Simulation on the lanes of the group: see 6.1 | `warp_size` reported as the simulated `W`, see 6.1 |
| Interpreter (IR) | same algorithm as Native on the interpreter's effect-based threads (`Sarek_ir_interp_eval.ml:957-975`) | idem |

### 6.1 Native and Interpreter (slow sequential version, mandatory)

Both execute a block as cooperating threads released by barriers (Native:
`Sarek_cpu_runtime`/`Sarek_cpu_runtime_exec` BSP path; Interpreter: `Barrier`/
`Warp_barrier` effects). Algorithm per shuffle call, with a per-block exchange array
`xchg` of `n` typed slots (int32 as `Int32`, float32 as `Int32` bits or `float#`-free
`Bigarray.Array1` of float32, float64 as float64; **no `Obj`**, typed module):

1. each thread writes `v` to `xchg.(t)`;
2. block barrier (a warp barrier already maps to the block barrier in Native,
   `Sarek_native_intrinsics.ml:249-251`; this is a superset and valid because every
   thread of the block reaches the call, which the convergence rule guarantees);
3. each thread computes `s` with the section-4 table using the simulated `W` and reads
   `xchg.(base + s)`, `base = (t / W) * W`;
4. second barrier, so a following shuffle cannot overwrite a slot still being read
   (or double-buffered; the implementer picks, with a test).

`W` for Native/Interpreter is a launch-time parameter: default 32, overridable per
launch config and by a test hook (like `Matrix_specialise.subgroup_override_for_tests`),
so the GPU parity tests run Native at the GPU's `W`. `Native_plugin_base.ml:250`
reports `warp_size = 1` and `Interpreter_plugin_base.ml:247` reports 32 only when
parallel: both must report the simulated `W` consistently (an `api-change` of reported
capability; check consumers that test `warp_size = 1`). The PPX native generator must
treat a kernel with `uses_warp_collective` as a barrier kernel (not the simple/fission
modes; `Sarek_native_gen*.ml`, `has_barriers` selection in `Sarek_cpu_runtime.ml:144-172`).
Cost: 2 barriers per call; documented as slow, correctness oracle only.

### 6.5 Lane-mapping self-test (Vulkan, OpenCL, and at least one run on the others)

On first use of a shuffle on a device, run (once per device, cached with the
capabilities) a probe kernel: each thread shuffles its own linear thread id from lane
`(i+1) mod W`, and a shuffle-xor by 1; compare with `t` computed arithmetically, for
1-D and 2-D block shapes with `bx` in `{8, 32, 64}`. Mismatch: the device refuses
shuffle kernels with block shapes that fail, naming the shape. This turns the
"lane = linear thread id mod W" assumption into a measured one (same practice as the
coopmat support probes).

## 7. Convergence rule (kept, and re-enabled in source)

`WarpConvergence` stays; a shuffle in diverged flow remains the located error
`Warp_collective_in_diverged_flow`. Because the primitives are no longer refused
earlier, `sarek/tests/negative/test_warp_diverged.ml` is changed to assert message 2
again (its own header says so), and `test_warp_shuffle_refused.ml` and
`test_warp_open_std_refused.ml` are rewritten: the first becomes a positive
compile test (converged shuffle accepted), the second keeps testing refusal for a
still-unlowered warp primitive (`warp_ballot`). `test_unlowered.ml`
("every warp primitive is refused") is restricted to the still-unlowered set.
`test_convergence.ml` unit cases stay and gain: shuffle after uniform `if` accepted,
inside a loop with varying bound refused, after varying early return refused, inside
a helper called from diverged flow refused (transitive rule of `convergence-semantics.md` §1).
The Rocq/Quint model (`formal/`) is not changed unless its warp row changes; the brief
asserts it does not, to be checked by the implementer (differential test between
production and extracted model stays green).

## 8. Risks

- **R1 Lane mapping on Vulkan/OpenCL.** The Vulkan and OpenCL specs do not tie the
  subgroup lane to the linear local id. Mitigation: self-test 6.5, launch check 4.2,
  refusal with the failing shape named.
- **R2 IR passes reorder or rewrite threads.** A shuffle marked `Pure` can be CSE'd,
  hoisted or sunk across a conditional by `Sarek_ir_unroll`, `Sarek_ir_softmath`,
  `Sarek_ir_for_bound`, `Sarek_ir_inline_vec`, `Fusion_epilogue`, `Fusion_horizontal`,
  `Launch_window`; and `Index_slice` / `Coverage_affine` / horizontal fusion change
  *which threads run or the block shape*, which changes warp membership. Mitigation:
  a convergent marker the passes respect, and each of these passes refuses (leaves
  unchanged) a kernel with `uses_warp_collective`; one test per pass.
- **R3 Out-of-range semantics differ across vendors** (low 5 bits of delta in PTX,
  undefined in Metal/Vulkan/OpenCL). Mitigation: guards plus boundary tests on every
  available device; any device whose measured result disagrees with Native fails the
  parity test.
- **R4 Variable `W`.** AMD Vulkan may run subgroups of 32 or 64; Intel 8..32; OpenCL
  `warp_size` is wrongly hard-coded. Mitigation: guards use runtime built-ins; fix
  OpenCL capability; parity test parameterised on `W`.
- **R5 Native perf and memory.** 2 barriers per call and an `n`-slot array per block.
  Accepted (oracle only); a test with a block of 1024 threads must finish in bounded time.
- **R6 Hardware coverage.** Available: tartarus (HIP gfx1100, W=32; Vulkan RADV; CUDA
  only through ZLUDA; OpenCL through whatever platforms exist). Metal: **ladon only in
  an owner-announced window** (project rule: no use of ladon otherwise; agent prompts
  must forbid it). Until then Metal and WGSL are golden-text only and marked
  *unverified*. No stress of the iGPU (`hip:1`). Vulkan runs: scan the kernel log
  (`dmesg`) before and after, as required by the 2026-10-12 Vulkan incident.
- **R7 PTX emitter is owned by another task** (`Sarek_ir_intrinsic_dispatch.ml` header):
  coordinate; if a conflict is open, PTX ships in a follow-up and the coverage table
  says so (acceptance AC-1 is then "7 of 8" with the PTX row explicit).

## 9. Trust boundary

- The kernel source (`[%kernel]`) is trusted code of the library user; no untrusted
  parsing is added.
- Out-of-range lane/delta/mask values come from kernel data and are untrusted at run
  time: the guards make every value defined; no out-of-bounds read of `xchg` is
  possible (index masked into `[base, base+W)` and `base + W <= n` guaranteed by 4.2).
- Device capability strings (`CL_DEVICE_EXTENSIONS`, Vulkan feature bits) come from
  drivers: treated as hints only; the lane-mapping self-test (6.5) is what authorises
  use. No new FFI surface beyond reading two more fields in existing queries.
- Emitted source includes only fixed spellings and integer expressions derived from
  typed IR values; no user string reaches the generated text.

## 10. Acceptance criteria (measurable)

- **AC-1** Coverage table (section 11) filled; every row is *tested* (command and
  result), *golden only* with the reason, or *refused* with the exact error text.
  PTX/CUDA C/HIP/Vulkan/Native/Interpreter are *tested* on tartarus (CUDA via ZLUDA if
  it runs, else golden + reason); Metal/WGSL may be golden-only (R6).
- **AC-2** Parity: for each of the four primitives, a kernel on `int32` and on
  `float32` over every boundary value of 4.3, `W` in `{32}` on all devices and
  `{8, 16, 64}` via the Native/Interpreter test hook, one block of `4*W` threads and a
  2-D block `(8, 4*W/8)`: GPU output equals Native output equals Interpreter output
  **bit for bit**. Run on every backend available in the CI-less local run, and the
  Native-vs-Interpreter part in `dune test` (no GPU).
- **AC-3** Downstream-shaped kernel: 8-lane xor-tree sum of `float32` per column (masks
  4, 2, 1), result compared with a barrier-and-shared-memory reference on Native,
  Interpreter and HIP; identical (sums are done in the same order, so equality is
  exact; if the order differs the test uses the documented tolerance and says so).
- **AC-4** Negative tests compile-fail with the located message: shuffle under
  varying `if`; under varying `while`/`for` bound; after varying early return; in a
  helper called from diverged flow; wrong arity; non-`int32` lane operand;
  unsupported value type. Positive compile test: converged shuffle accepted.
- **AC-5** Golden codegen for each source backend (CUDA, HIP, OpenCL, Vulkan GLSL,
  Metal, WGSL, PTX): per primitive and per value type, text equals the stored golden;
  `test_intrinsic_fallback_all` uses another unknown name (a deliberately unlowered
  probe) and still passes on all backends.
- **AC-6** Capability tests: faked capabilities without the OpenCL extension / the
  Vulkan shuffle bits / the WebGPU feature produce the exact located refusal; with
  them, generation succeeds. OpenCL `warp_size` is no longer a constant 32.
- **AC-7** Launch check: block size not a multiple of `W` for a shuffle kernel gives
  `Invalid_launch`; a kernel without shuffles is unaffected (existing tests unchanged).
- **AC-8** Pass-safety tests (R2): for each listed pass, a kernel with a shuffle is
  left unmoved/unfused/unsliced, shown by IR equality before/after the pass.
- **AC-9** Perf evidence (not a gate on merge, but required before the downstream engine claims
  anything): micro-benchmark on gfx1100, shuffle 8-lane reduction vs shared+2 barriers,
  command, machine, date, raw output location recorded (project rule). No performance
  number appears in docs without it.
- **AC-10** `dune build @fmt`, `dune build`, `dune test` green; no existing test
  changed except those named in section 7; new files <= 500 lines, functions <= 50.

## 11. Coverage table (filled at implementation, 2026-10-17)

Machine for the device runs: tartarus (Ryzen 9 7950X, RX 7900 XTX gfx1100 only,
`HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, `RUSTICL_ENABLE=radeonsi:0`,
`SPOC_DISABLE_VULKAN=1`, under a GPU lock file (`flock`) and
`scripts/gpu-kernel-log-gate.sh`; the kernel log stayed clean). Raw outputs are
kept outside the repository, in `/tmp/ws-evidence/` on tartarus. The load
average read 2 to 4 during the runs (other sessions), which does not matter for
the pass/fail parity runs and is why the timing is marked provisional.

| Backend | Lowered | Capability gate | Golden | Device test | Parity vs Native | Status |
|---|---|---|---|---|---|---|
| CUDA C (NVRTC) | `__shfl*_sync`, `warpSize`, clamped delta/mask (shared generator with HIP) | `Warp_shuffle` listed (sm_30 and above) | `warp_shuffle_cuda.expected` | none (no NVIDIA device; the `#else` branch is never compiled) | by text | golden only, unverified on NVRTC |
| PTX | `shfl.sync.{idx,down,up,bfly}.b32`, `laneid`, `WARP_SZ`, f64 as two halves, idx masked and relative modes clamped | listed | `warp_shuffle_ptx.expected` | ZLUDA on gfx1100: 24/24 shapes and types (`test_warp_shuffle_parity`), reduce (`test_warp_shuffle_reduce`) | bit for bit | tested through ZLUDA; found and fixed: ZLUDA returns the own value for `idx` operands >= 32, hence the explicit `and 31` |
| HIP | `__shfl*` with `width = warpSize`, clamped | listed | same source as CUDA C | gfx1100 (W = 32): 24/24, reduce | bit for bit | tested; W = 64 not run (no device) |
| OpenCL | `sub_group_shuffle` with a computed source lane, `-cl-std=CL3.0` | `cl_khr_subgroup_shuffle` and the measured lane mapping (`Opencl_warp_probe`), `warp_size` measured | `warp_shuffle_opencl.expected` (also syntax-checked by clang `-cl-std=CL3.0`) | RX 7900 XTX under rusticl, W = 64: 16/16 (no fp64 on that device), reduce | bit for bit | tested at W = 64; pocl and llvmpipe CPU refuse (8x8 lane mapping fails) |
| Vulkan | `subgroupShuffle` with a computed source lane, f64 as two halves, Vulkan 1.1 environment | `BASIC` and `SHUFFLE` for compute (`subgroup_shuffle_supported`) | `warp_shuffle_glsl.expected`, `glslangValidator --target-env vulkan1.1`, `spirv-val` | none: Vulkan is not run on tartarus (owner rule) | by text | golden and offline validation only; lane mapping not probed |
| Metal | `simd_shuffle` with a computed source lane; attributes passed to helpers | listed | `warp_shuffle_metal.expected` | none (decision D7) | by text | golden only; a warp operation in a user helper is refused |
| WGSL | `subgroupShuffle`, `enable subgroups;`, `var<private>` lane and width | no WebGPU runtime device in the tree | `warp_shuffle_wgsl.expected` | none | by text | golden only, unverified (naga 30 does not implement `subgroups` yet); no f64 |
| Native | `Sarek_cpu_runtime.Warp` (exchange array, two barriers), width 32 by default | `Warp_shuffle` listed | n/a | `test_warp_shuffle_parity` W in {8, 16, 32, 64}, `test_warp_shuffle_reduce` | oracle | tested |
| Interpreter | same algorithm on the effect-based threads | listed | n/a | same tests | equal to Native | tested |

Provisional timing (AC-9, not a gate), `bench_warp_shuffle` on gfx1100 HIP,
2026-10-17, 4096 dependent 8-lane float32 reductions per thread, 1,048,576
threads, median of 7 alternating pairs, machine loaded: 3 shuffles 5.55 ms,
shared memory with 2 barriers 11.65 ms (2.10x). No performance claim for the downstream engine
rests on this number; it must measure its own kernel in a quiet window.

Deviations from the brief, all recorded in the PR:
- the launch error is the existing `Execute_error.Invalid_dimensions` (there is
  no `Invalid_launch`);
- OpenCL, Vulkan, Metal and WGSL lower `down`, `up` and `xor` as index shuffles
  with a computed source lane (no `_relative` extension, no relative bit);
- the lane-mapping probe (section 6.5) is implemented for OpenCL only; Vulkan is
  not run, and CUDA and HIP rely on the architecture;
- the PPX typing is an overload in the typer (decision D1, option a);
- rule H-2 of the convergence spec (a barrier or collective hidden in a helper)
  was documented but not implemented; it is now, for barriers and collectives.

## 12. Documentation

`sarek/sarek/BSP.md` (replace "not available yet" for the four primitives, keep it for
the others), `specs/convergence-semantics.md` §7 and §8 (rows and S1 note),
`docs/optimization/opt-expressivity-gaps.md` (gap 1 status, the update note at line ~82),
`sarek-*/README`/backend capability docs (extension requirements, refusal texts),
`CHANGES.md` (api-change entry: new callable primitives, capability fields, launch
check, reported `warp_size` of Native/Interpreter/OpenCL), `COVERAGE.md` if it lists
intrinsics, KB (`kb/backends`, `kb/sarek`) via the usual post-merge update. A user
guide paragraph: the exact table of 4.3, the multiple-of-`W` rule, the segment idiom
with `xor`.

## 13. Decisions (owner, validated 2026-10-17)

Options considered are in the earlier versions of this file (Git history). Binding decisions:

- **D1 Typing.** `int32`, `float32`, `float64`. Overloaded on the value type if the Sarek
  typer allows it (precedent to check: how `abs`/`min` are typed); otherwise type-suffixed
  names (`warp_shuffle_f32`...).
- **D2 Segment width.** None in v1. Sub-warp groups use `xor` with masks below a power of two.
- **D3 Partial warps.** A kernel using a shuffle whose block size is not a multiple of the
  warp width is refused at launch (`Invalid_launch`).
- **D4 `warp_size` and `lane_id`.** Exposed in this task (portable tree reductions need `W`).
  They leave the unlowered set and follow the same per-backend lowering and tests.
- **D5 OpenCL/WGSL without the extension.** Clear located refusal in v1. Shared-memory
  emulation follows in a later task. Owner requirement: the Sarek compiler must eventually
  handle shuffle everywhere, Native included, if necessary through a slow sequential version.
- **D6 PTX.** Same PR, unless another task touching the PTX emitter conflicts (then a
  follow-up, with the PTX row explicit in the coverage table).
- **D7 Metal.** Golden tests only. No execution on ladon except in a window announced by the owner.

## 14. Out of scope

`warp_vote_all/any`, `warp_ballot`, `warp_active_mask`, `warp_vote_*` family, warp-scope supersteps (S1), `width`-segmented shuffles (D2: none in v1), shuffle
of `bool`/vector/record values, subgroup reductions/scans/arithmetic
(`subgroupAdd`...), cooperative groups, non-full masks, any downstream kernel change (the engine
adopts the primitives in its own branch after this merges, with its own perf-vs-llama
report), CI workflow changes.

## 15. Process notes

SPOC work only in this worktree; never build or run gates in `~/dev/SPOC`. Roster
pipeline in the worktree after validation; merge to SPOC `main` when the full pipeline
passes and CI is green. Shared GPU: record load, alternate A/B pairs, no conclusive
perf numbers outside an owner-announced quiet window. Every new `let%sarek_intrinsic`
or capability field gets a test; no `Obj`; typed modules/functors for the per-type
exchange buffers.
