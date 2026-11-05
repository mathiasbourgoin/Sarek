## 2026-10

### Fixed

- Code generation of valid OCaml kernels (found by Katra, review of 2026-10-17): a name rebound in the
  same scope (`let s = s +. 1.0 in`, two `let t` in one block) was emitted as a C redeclaration on
  CUDA/HIP, OpenCL and Metal (and, for two `let t`, on GLSL and WGSL); a variable named like a keyword,
  type, qualifier or built-in of the target (`half`, `local`, `kernel`, `uint`, `float4`, `max`) or
  not an identifier (`x'`) broke the device compiler or hid a built-in. `Sarek_ir_names.legalize` now
  renames such binders to `<name>_<k>` (`v_<name>_<k>` where a prefix family reserves `<name>_<k>`)
  at the entry of each generator, references following by identity (`var_id`) first and by name second, so a later
  statement that reaches an outer binder of the same name is not redirected to the renamed inner one;
  a kernel without a clash and without a name of the reserved lists is returned unchanged and its
  source is byte-identical (every golden unchanged); the lists also hold names that compiled before
  (`length`, `step`, `dot`, the `get_`, `half_` and `quad_` families), whose kernels get a new source.
  A `let` in a helper that rebinds a helper parameter is renamed on every target. GLSL and WGSL keep their own
  escapers for reserved words and for a local shadowing a scalar parameter. Not covered: `SNative`
  kernels (opaque source), built-in function names on GLSL and WGSL, cooperative-matrix fragment
  names, the names of record and variant types. Tests: `sarek/tests/codegen_golden/test_name_legalization.ml` and `test_name_scope.ml` (every backend; compiled with
  `clang -x cl` and the Metal runtime compiler when present, which reject the pre-fix spelling).
- Launch cost regression since the warp-shuffle PR: `Execute.check_launch_warp` and
  `Matrix_specialise.decide` folded the whole IR (`kernel_uses_shuffle`, `kernel_uses_warp_ops`) at
  every launch, which cut Katra's decode throughput by 42 to 53 %. Both verdicts are now recorded once
  per IR (a slot of its `Ir_memo` record, and a field of the matrix-specialise entry). Same refusals
  and messages. Test: `sarek/execute/test/test_launch_check_cost.ml` (1000 launches must cost under
  50 whole-IR folds; the old code cost 878). Rule: anything computed from the IR at launch is
  memoised per kernel.

### Added

- `float16` in the PTX backend (#529), which used to refuse every kernel that used it. f16 stays a
  storage type: vectors, locals, shared and local arrays, selects, inlined helpers and the two
  conversions emit a `%h<N>` register class (`.reg .b16`), `ld`/`st`/`mov`/`selp` on `.b16`,
  `cvt.f32.f16` (exact) and `cvt.rn.f16.f32` (round to nearest even, the interpreter's rounding);
  int32/int64/f64 conversions go through f32. No native `.f16` instruction is emitted, so the same
  module assembles for every target from sm_50 (native f16 arithmetic needs sm_53). The register
  class is decoded in one place (`Sarek_ir_ptx_types.reg_class`); an unknown prefix, an f16 operand
  of an operator, intrinsic or index, an f32 stored into an f16 array (or the reverse) and a mov
  between an f16 and a non-f16 register are now located `Ptx_codegen_error`s instead of falling
  into an integer default. Still refused: scalar `float16` kernel parameters and `float16` record
  fields. Evidence: `test_ptx_f16` (structure, no native f16, identical body sm_50..sm_90), the
  `ptxas` gate of `test_ptx_snapshot` (sm_52, sm_61, sm_75, sm_90; CI), and
  `test_ptx_f16_e2e` over all 63488 finite binary16 inputs. Executed on ZLUDA/AMD (RX 7900 XTX)
  only: raw round trip, f16 -> int32 -> f16 and int32 -> f16 agree with the interpreter on every
  input; `f16(x*1.1)` and `f16(f16(x*1.1)+1000)` differ on 2913 and 620 inputs because ZLUDA's
  AMDGPU lowering fuses the multiply into the narrowing. NVIDIA execution is pending
  (`docs/fp-contraction-policy.md`).
- `Spoc_core.Mapped_registration`: the capability to copy straight from registered file-backed pages
  (a read-only shared mapping of a file registered with `Transfer.register_host`), asked of the
  library instead of deduced by each consumer. `static_verdict dev` decides from the backend (it registers and
  copies nothing; HIP and CUDA read one device attribute): Vulkan is `Refused (Kernel_hazard _)` whatever `SPOC_VULKAN_HOST_IMPORT*` say
  (registration is never attempted there, Sarek #503/#504); Metal and the CPU devices (Native,
  Interpreter, CPU OpenCL) are `Not_needed` (copy from the mapping); GPU OpenCL and unknown backends are `Refused (Not_measured _)`;
  HIP and CUDA are `Refused Integrated_gpu` / `Device_kind_unknown` or ask for the self-test.
  `probe dev ~map ~reference` runs that self-test: two disjoint windows are registered, copied to the
  device and back, compared with an independent read of the file (`hipHostRegister` returns success
  on `PROT_NONE`, tmpfs and overlapping ranges, so the return code decides nothing), timed against
  the staged copy, and released; the verdict is `Usable | Not_needed | Refused | Too_slow |
  Inconclusive | Still_registered` (a window the probe could not release: keep the whole `map` mapped, and neither probe nor register it again in this process), and `cacheable` is true only for what the self-test established (`Usable`, `Refused Wrong_bytes`). Additive: a new module, and
  `Spoc_framework.Shared_memory_query`, a hook through which HIP and CUDA say whether a device
  shares host memory (no field added to any existing record or signature). SPOC keeps no cache and
  no mapping code; the consumer owns both. Tests: `sarek/core/test/test_mapped_registration.ml`
  (no GPU), `sarek-metal/test/test_metal_mapped_registration.ml` (`@e2e-metal`).
- Launch cost gate in CI: `sarek/execute/test/test_launch_cost.ml` times 2000 launches through
  `Execute.run_vectors` on two probe backends (the immediate launch path, and one named "HIP" that
  takes the deferred launch-window path and is tried by the matrix-unit rewrite) and fails if a
  launch of a 20000-statement kernel costs more than 25 times a launch of a 1-statement kernel (a
  per-launch IR walk), or if a launch costs more than 8 units of an in-process reference loop (a
  fixed cost added to every launch). Idle, the figures are 1.1 times and 0.7 to 1.3 units; reverting
  `Execute` and `Matrix_specialise` to the #522 code gives 830 to 1540 times, and a fold in only the
  matrix decision (invisible to an immediate-path probe) about 570 times. See
  `docs/launch-cost-gate.md`.

### Changed

- Convergence: the strict variance analysis introduced for warp shuffles now also covers block
  barriers and supersteps. A barrier is refused when the thread-varying condition reaches it through a
  helper parameter that the caller makes varying, a helper result that reads a thread index, the
  value of a thread-varying `if`/`match`, or a `mut` variable assigned under thread-varying control
  flow or a thread-varying value (`Sarek_convergence.check_kernel`; negative tests
  `test_barrier_diverged_param`, `_result`, `_mut`, and `test_convergence` "strict barrier"). Before
  this change these four kernels were accepted and would deadlock or give undefined results on a
  GPU. Kernels accepted before and refused now: none among the repository's kernels (tests,
  examples, benchmarks) nor Katra's `src/kernels` (compiled with the new analysis); a kernel that
  relies on one of the four routes to hide a barrier under a varying test must make the test
  uniform (hoist the barrier out of the branch).
  Limits of the stricter analysis for barriers: the variance of a `mut` is flow-insensitive, so a
  `mut` used as a uniform loop counter for a barrier loop and later assigned a thread-varying value
  is refused (use a separate `mut`); a barrier after a tuple `let` that destructures a thread-varying
  component is refused (already so before this change); the strict pass is exponential in the length of
  helper call chains (a chain of 22 helpers takes about 5 s to compile, against 0.2 s before; nested
  `let mut` scans are memoised: a kernel of 22 nested mutables compile in 0.2 s and a 22-helper chain in 3 s); an inline `let` whose bound value varies but whose body does not still counts as varying in the legacy pass (so `let n = (let t = thread_idx_x in ...; 8l) in` before a barrier loop is refused, as before; the helper form is accepted); helpers resolve names lexically in the strict pass (a helper
  redefined after use no longer hides a barrier).

### Added

- Warp shuffles: `warp_shuffle`, `warp_shuffle_down`, `warp_shuffle_up` and `warp_shuffle_xor`
  are callable from `[%kernel]` code on `int32`, `float32` and `float64` values (one name,
  overloaded by the typer; the IR carries the type in the name, `warp_shuffle_down_f32`), with
  `warp_size` and `lane_id` (`int32`). One contract on every backend, specified independently of
  any vendor (`Sarek_warp.source_lane`, `sarek/sarek/BSP.md`): lane `i` reads lane `x land (W-1)`
  (`warp_shuffle`), `i + d`, `i - d` or `i lxor m`, and its own value when the source is out of
  range (`d` or `m` negative, or the source outside `[0, W)`). Every lane of the warp takes part;
  a shuffle in diverged flow is a compile-time error. Coverage, one row per backend: CUDA C
  (NVRTC) `__shfl*_sync`, source tested as text only; PTX `shfl.sync` (two shuffles for a
  `float64`), run through ZLUDA on gfx1100; HIP `__shfl*` with `width = warpSize`, run on gfx1100;
  OpenCL `sub_group_shuffle` (extension `cl_khr_subgroup_shuffle`, built at the OpenCL C level of the
  device), run on the RX 7900 XTX under rusticl at W = 64; Vulkan GLSL `subgroupShuffle`, validated
  with `glslangValidator`/`spirv-val` but not run (Vulkan stays off the shared workstation); Metal
  `simd_shuffle` and WGSL `subgroupShuffle`, text only (decision D7 of the plan, WGSL is unverified:
  the validators in the tree do not know the `subgroups` extension yet); Native and the Interpreter
  simulate a warp (default 32 lanes, `Sarek_warp.with_width` as the test hook) with a per-block
  exchange array and two barriers per call. Device results equal Native and Interpreter bit for bit
  over the boundary operands `{0, 1, W-1, W, W+1, -1, 2W+1}`, for a 1-D and a 2-D block
  (`test_warp_shuffle_parity`), and an 8-lane xor-tree float32 sum equals the same tree through shared
  memory (`test_warp_shuffle_reduce`). The up, down and xor flavours are emitted as index shuffles with
  a computed source lane on OpenCL, Vulkan, Metal and WGSL, so only the base shuffle extension is
  needed. A device without the capability refuses a kernel that uses a warp operation with the
  capability model's message (`warp-shuffle unavailable`); shared-memory emulation on those devices
  is planned. **Launch rule:** a kernel that shuffles is refused with
  `Execute_error.Invalid_dimensions` when its block is not a multiple of the device's warp width
  (`Execute.check_launch_warp`; Native and the Interpreter use the simulated width). **API change for
  out-of-tree code:** `Sarek_ir_analysis.feature` gains `Warp_shuffle` (listed in a device's
  `device_features`; every in-tree backend is updated); `Sarek_core_primitives.purity` gains
  `Convergent` (the shuffles are no longer `Pure`) and the table gains `lane_id`;
  `Sarek_ir_intrinsic_dispatch.spec` gains `warp_dialect`; `Sarek_error.unlowered_reason` loses
  `Warp_size` and `Sarek_error.error` gains `Warp_shuffle_value_type`;
  `Sarek_native_intrinsics.gen_intrinsic_fun` gains `?result_ty`; `Sarek_ir_ptx_expr.intrinsic_category`
  gains `Warp`; the Vulkan device record gains `subgroup_shuffle`, the OpenCL one `version` and
  `supports_subgroup_shuffle`. The user-facing names `warp_shuffle*`, `warp_size` and `lane_id` now
  resolve to the intrinsics, as `block_barrier` does: a kernel-local helper with one of these names no
  longer shadows it. A kernel that uses a shuffle is a barrier kernel for the CPU runtimes, and
  no IR pass that moves threads touches one: horizontal and epilogue fusion, the launch window, the
  matrix rewrite, `Index_slice` and `Coverage_affine` refuse such a kernel (`test_warp_pass_safety`).
  The convergence check refuses a shuffle reached through a helper parameter that the caller makes
  thread-varying, a helper result that reads a thread index, the value of a thread-varying
  `if`/`match`, or a `mut` variable assigned under thread-varying control flow or a thread-varying
  value (`Sarek_convergence`, a strict variance analysis applied to warp collectives, barriers and supersteps;
  `test_warp_diverged_param`, `_result`, `_mut`, `_if_value` and `test_convergence` "strict").
  **Known limits:** reads of memory at a uniform index count as
  uniform (specs/convergence-semantics.md section 3, race freedom). Still accepted for a warp
  shuffle or a barrier (found by the independent review of 24fd1b40, follow-up work): a helper selected through a
  function-valued `if` (the check runs before defunctionalisation), a recursive helper whose
  recursive call makes a parameter varying, variance stored through a local array or record, a `mut`
  assigned inside a local helper, and a call chain of more than 8 helpers; the strict analysis is
  exponential in the length of helper call chains (the `let mut` scans are memoised) and skipped for kernels without a barrier, superstep or warp collective;
  it rejects a tuple `let` whose scrutinee is varying even if the shuffle does not use it, and a shuffle in a `while` condition whose value counts as thread-varying (the shuffle result itself is varying, so even a loop on a uniform value is refused); a shuffling helper passed as an argument to a higher-order helper is still accepted (same root cause as the function-valued `if`); a `let s = ... s ...` that rebinds
  a name in the same scope (including a kernel parameter's name) still emits a C redefinition on the
  CUDA, HIP and OpenCL source backends (pre-existing, not specific to warps); Vulkan claims the feature only with `SPOC_VULKAN_SUBGROUP_SHUFFLE=1` (no lane-mapping
  probe, full subgroups not required); the OpenCL probe runs at device enumeration and measures the
  width of its own kernel. Consumers that read `Device.warp_size` (Katra's `Gdn.threads_for`) see the
  new values above.
  Tests: also `test_sarek_warp`, `test_warp_capability`, `test_opencl_warp_capability`,
  `test_vulkan_warp_capability`, `test_warp_shuffle_golden` (six goldens), `test_warp_shuffle_converged`,
  twelve new `make test_negative` cases. Docs: `sarek/sarek/BSP.md`, `gh-pages/docs/kernel_language.md`,
  the backend READMEs, `specs/convergence-semantics.md`, `docs/plans/warp-shuffle-intake-2026-10-17.md`.

- Launch bounds: `pragma ["sarek.max_threads N"]` around a kernel body declares an upper bound
  on the block size (`kern_launch_bounds : launch_bounds option` in `Sarek_ir_types.kernel`;
  `Sarek_ir_types.combine_launch_bounds`). Default unchanged: no pragma, byte-identical source.
  Coverage: HIP and CUDA-C `__launch_bounds__(N)`; CUDA PTX `.maxntid` and OpenCL
  `reqd_work_group_size`, both from the launch block (`?block` is now used by
  `Sarek_ir_ptx.generate*` and by the OpenCL plugin) when it respects `N`; Metal
  `[[max_total_threads_per_threadgroup(N)]]` (generation tested, not compiled on a Mac);
  Vulkan, WebGPU, Native, Interpreter: no source change. A launch whose block exceeds `N`
  is refused on every backend with `Execute_error.Invalid_dimensions`
  (`Execute.check_launch_bounds`). Fused kernels take the minimum bound. **API change for
  out-of-tree code:** `Sarek_ir_types.kernel` gains the field `kern_launch_bounds` (use
  `{default_kernel with ...}`), `Sarek_ir_opencl.generate_with_types` gains `?block`, and
  `Sarek_ir_ptx.generate` / `generate_with_types` gain `?block`. Tests: `test_launch_bounds`,
  `make test_negative` (four refusals). Docs: `kernel_language.md`, Launch bounds.

### Changed

- Native: PPX-generated kernels read and write float32/int32 vector parameters through the Bigarray
  directly (`Sarek.Sarek_native_fast`) instead of the accessor object, and `Vector.ensure_cpu_sync`
  runs once per vector per launch instead of per element. Bit-identical results (digest test
  `test_native_typed_access_exact`, recorded with the previous implementation); same bounds error
  (an out-of-range index still reaches `Vector.get`/`Vector.set` and raises their `Invalid_argument`);
  a store still marks the device copy stale as `Vector.set` does. Other element types and
  non-Bigarray storage keep the accessor path. Not changed: superstep handling, arithmetic.
  `bench_native_matvec` native, median of 15, alternating, tartarus load about 1: row 55.5 ms to
  29.5 ms, wave 189 ms to 140 ms, copy 4.0 ms to 2.9 ms; checksums identical.
  One observable difference: a float32/int32 vector argument with Bigarray host storage in `Stale_CPU` is now
  downloaded when the launch arguments are built, even if the kernel never reads it (it becomes
  `Both`); vectors of other element types or storage are untouched; the native
  argument record is valid for the launch that built it only.
- `warp_size` as reported by a device: Native and the Interpreter report the simulated warp width
  (32 by default; the Interpreter reported 32 only when running in parallel and Native reported 1),
  and OpenCL reports the sub-group size measured by a probe kernel (it reported the constant 32; a
  device without sub-groups now reports 1). Nothing in the tree compared them with the old values.
- Convergence rule H-2 of `specs/convergence-semantics.md` is now implemented: a call reached in
  diverged flow to a kernel-local or module helper that contains a synchronisation point
  (transitively), a block barrier or a warp collective, is rejected at the call. Before, a barrier
  hidden in a helper escaped the divergence check (scenario G-9 of the spec compiled). New
  `make test_negative` cases `test_barrier_helper_diverged` and `test_warp_diverged_helper`.
- Vulkan compiles a shader that uses subgroup operations for the Vulkan 1.1 environment (SPIR-V
  1.3), through shaderc and through `glslangValidator`; every other shader keeps the default
  environment.

- `Transfer.create_stream ?non_blocking:bool` (default `false`, unchanged): `~non_blocking:true`
  makes a stream the device's default stream does not wait for (HIP `hipStreamNonBlocking`,
  CUDA `CU_STREAM_NON_BLOCKING`), so a copy stream can overlap kernels on the default stream;
  order streams with `record_event` / `stream_wait_event`. The implicit paths stay correct: a
  host read-back (`to_cpu`, `sync`, migration), the implicit upload (`to_device`), a
  default-stream launch (`Execute.run`, `run_vectors`, `run_source`, SoA launch), and default-stream
  ranged copies and `stage_to_device` first wait on the host (`Transfer.settle_vector`) for any
  non-blocking stream still referencing the vector; launches and copies on an explicit stream are
  never settled, ordering is the caller's through events. Host buffers (a pinned buffer used by
  a pending non-blocking copy) are not tracked by the settle: ordering them is the caller's.
  **Breaking for out-of-tree backends:** `Framework_sig.BACKEND.Stream` (and `PLUGIN_BASE`) gain
  `create_non_blocking : Device.t -> t` (one line: `let create_non_blocking = create` when the
  backend's queues are already independent). Coverage: HIP and CUDA (C and PTX) implemented;
  OpenCL, Metal, Vulkan: same as `create` (independent queues); Native, Interpreter: unit
  streams, synchronous; WebGPU: stub, unchanged. Tests: `test_nonblocking_streams`
  (`runtest` on Native and Interpreter; GPU devices by argument), `test_hip_nonblocking`.
- `Device.is_gpu` is true for Metal devices (it listed CUDA, OpenCL, Vulkan and HIP only), so a
  selection by GPU predicate finds the Metal GPU on a Mac (#516). Test: `test_device` checks
  the frameworks of the default `Device.init` list.
- Vulkan `alloc_pinned` no longer imports host memory: it allocates through the driver
  (`vkAllocateMemory` from a HOST_VISIBLE | HOST_COHERENT type that is not DEVICE_LOCAL,
  HOST_CACHED when the device has one) and maps it for the life of the block, so it works on
  every device, AMD included, with no opt-in and no `VK_EXT_external_memory_host`. No host page
  reaches the driver, hence no userptr object (the #503 path). `register_host` is unchanged:
  opt-in, refused on AMD. A pinned block now belongs to its device and is unmapped with it
  (a late `free_pinned` still succeeds). Tests: `test_vulkan_host_import_policy`,
  `test_vulkan_ranged_copies` and `test_vulkan_host_lifecycle` run the pinned path on the
  device. Measured on the RX 7900 XTX, 64 MiB: 21.5-22.3 GB/s host-to-device and 21.8-22.0 GB/s
  device-to-host through a pinned block, against 5.5 and 2.5 GB/s through staging
  (`bench_vulkan_pinned_copy`, machine under load). `pread` with `O_DIRECT` into the block works
  on RADV (`test_vulkan_host_lifecycle`). Allocation can now fail with an out-of-memory error
  from `vkAllocateMemory` instead of a `Feature_not_supported` refusal, and `stage_to_device`
  works on Vulkan where it was refused.
- **Breaking.** `SAREK_OPT_DISABLE` and `SAREK_EXPLAIN` replace five switches
  (`specs/optimisation-policy.md` 3.4-3.5, milestone M3, `gh-pages/docs/environment.md`).
  Environment variables only restrict: nothing in the environment grants precision or turns
  on an optimisation. Both are read once, at program start. **Removed outright, no alias and
  no warning** (setting them now does nothing; `sarek/tests/policy_env` runs a process with
  all five set and checks the defaults hold):

  | Removed | Replacement |
  |---|---|
  | `SAREK_LAUNCH_WINDOW=0` | `SAREK_OPT_DISABLE=window` |
  | `SAREK_LAUNCH_EPILOGUE=0` | `SAREK_OPT_DISABLE=epilogue` |
  | `SAREK_MATRIX_REWRITE=off` | `SAREK_OPT_DISABLE=matrix` |
  | `SAREK_MATRIX_REWRITE=exact` | `SAREK_OPT_DISABLE=automatic-float-rewrite` (int32 still rewritten; unlike the old mode it does not stop an explicit `Sarek_matrix.run`, which never needed the consent) |
  | `SAREK_AUTO_UNROLL=0` | `SAREK_OPT_DISABLE=unroll` |
  | `SAREK_MATRIX_REPORT=1` | `SAREK_EXPLAIN=1` |
  | `Matrix_specialise.mode`, `set_mode`, `parse_mode`, `mode_name`, type `mode`; `Sarek_matrix.mode`, `set_mode`, type `mode` | `Sarek_policy.restrict` (one-way, process-wide) |
  | `Matrix_specialise.set_reporter`, type `report` | `Sarek_policy.set_reporter` (receives `Decision` records) |

  `Matrix_specialise.lowering_for` loses its `~mode` argument and `specialised` gains
  `evidence`. `Execute.Window.set_enabled`, `set_backend_enabled` and `set_epilogue_backend`
  stay. `Sarek_policy.evidence`'s `Granted` now carries `{grants; reason; lowering}`,
  `Sarek_policy.record`'s `device` field is `target`, and `set_reporter` takes `event`s
  (`Decision of record | Gate | Ignored_grant`).
  `SAREK_OPT_DISABLE` is a comma list (any case, spaces and empty items ignored; an unknown
  item prints one warning line on stderr and is ignored): `window`, `epilogue`, `matrix`,
  `narrowing-proof`, `unroll` restrict that optimisation; `accumulation-order`,
  `operand-rounding-f16|bf16|bf16x3`, `single-rounding`, `cross-kernel-contraction`,
  `automatic-float-rewrite` revoke that grant for the whole process (the program's
  `accept_*`/`grant` call still succeeds and is recorded, but no budget holds it).
  `SAREK_EXPLAIN=1` writes one stderr line per **distinct** (kernel, target, optimisation,
  decision) for every decision point (a decision that flips back is not printed again):
  window (per launch), epilogue, the matrix rewrite with its lowering, the explicit coopmat
  gate of the device backends (its own `coopmat-gate` label, not an optimisation and not a
  `SAREK_OPT_DISABLE` item), the narrowing proof and unroll (per kernel generation), and a
  program `Kernel` grant on a generated (fused, derived, static-fusion) name or a revoked grant
  (`sarek-explain <name> - policy refused grant-on-generated-name|grant-revoked <grant>`). The
  target is `<framework>:<device id>` for window, matrix and the gate (`HIP:0`), `<framework>`
  for the epilogue, `gen:<generator>` for code generation (`gen:cuda-c`, `gen:opencl`,
  `gen:metal`, `gen:glsl`); a float matrix line ends with `lowering="..."` and its
  `reason="..."` holds only the program's reasons. Revoking `accumulation-order` (the closest
  to the old `SAREK_MATRIX_REWRITE=exact` for float paths) makes an explicit cooperative-matrix
  kernel fail at launch, with a refusal that says it is revoked by `SAREK_OPT_DISABLE`.
  Real lines from a run on an AMD RX 7900 XTX (HIP, `SAREK_EXPLAIN=1`; raw capture:
  `spoc-pipeline-logs/policy-impl/m3-r2-explain-sample-hip.txt` on the owner machine; the
  CUDA/PTX line is from ZLUDA on the same card, `m3-r2-e2e-zluda.txt`):

  ```text
  sarek-explain sarek_kern HIP:0 window applied exact
  sarek-explain sarek_kern HIP:0 matrix applied proof=int32 byte planes (u8*u8+s32->s32, 10 products), staged, 16x16x16 u8*u8+s32->s32 subgroup: bit-identical modulo 2^32
  sarek-explain sarek_kern HIP:0 matrix refused float products are rewritten automatically only after Sarek_coopmat.accept_automatic_float_rewrite (plus the accuracy opt-ins); Sarek_matrix.run does not need it
  sarek-explain sarek_kern HIP:0 matrix applied granted=automatic-float-rewrite,accumulation-order reason="test_matrix_optin" lowering="f16 operands (f16*f16+f32->f32), direct (2 waves/block), 16x16x16 f16*f16+f32->f32 subgroup"
  sarek-explain sarek_kern_sarek_mm_direct_w2 HIP:0 coopmat-gate applied granted=accumulation-order reason="test_matrix_optin"
  sarek-explain sarek_kern+sarek_kern HIP epilogue applied exact
  sarek-explain sarek_kern+sarek_kern OpenCL epilogue refused no grant cross-kernel-contraction (kernel or process)
  sarek-explain sarek_kern+sarek_kern CUDA/PTX epilogue applied granted=cross-kernel-contraction reason="test_epilogue_exact: measuring the contraction bound"
  sarek-explain sarek_kern gen:cuda-c narrowing-proof applied proof=4/4 narrowings proven exact
  sarek-explain sarek_kern gen:cuda-c narrowing-proof refused 0/4 narrowings proven exact, the barrier is kept
  sarek-explain f16_single_rounding_dequant gen:cuda-c narrowing-proof applied granted=single-rounding reason="test_hip_f16_single_rounding"
  sarek-explain sarek_kern_sarek_mm gen:cuda-c unroll applied exact
  ```

  Coverage (what ran where; none of it is a performance claim). Restriction scenarios, the
  parser and the explain golden: no device (`sarek/tests/policy_env`,
  `spoc/ir/test/test_sarek_policy_env.ml`), so they cover every backend's decision logic
  but not a backend's launch. Executed with `SAREK_EXPLAIN=1`: HIP (RX 7900 XTX: window,
  int32 and float matrix rewrite, epilogue, narrowing, single rounding, unroll), CUDA/PTX
  through ZLUDA (epilogue with the grant, window), OpenCL (epilogue refused without the
  grant, window). Native and Interpreter: only the CPU-device window refusal of the explain golden (a fake device); no run with `SAREK_EXPLAIN` on the real CPU backends.
  Vulkan: the sources are generated and the unroll hint is checked in the GLSL golden;
  the subset of the Vulkan tests that is not excluded (#503/#504) passed, no launch with
  `SAREK_EXPLAIN` was run. Metal: executed on an Apple M4 (2026-10-12): the Metal unit,
  execute, ir, policy_env and sarek-metal runtests pass; `SAREK_EXPLAIN=1
  test_epilogue_exact` prints `Metal epilogue applied exact` lines; `test_matrix_rewrite_metal`
  prints `Metal:1 matrix applied granted=accumulation-order reason=... lowering="f16 operands
  ..., staged, 8x8x8 ..."`, the integer refusal ("Metal has no integer simdgroup_matrix"),
  `coopmat-gate` and `gen:metal unroll` lines; with `SAREK_OPT_DISABLE=window,matrix` the
  rewrite assertions of `test_metal_matrix_rewrite` fail with "disabled (SAREK_OPT_DISABLE)",
  as expected (raw: `spoc-pipeline-logs/policy-impl/ladon/sarek-policy-m3-*.txt`, owner
  machine).

- Vulkan host-memory import (`alloc_pinned`, `register_host`, from #493) is now an explicit opt-in
  (`SPOC_VULKAN_HOST_IMPORT=1`) and is refused on AMD drivers even then
  (`Vulkan_host_import_policy`). On a refused device `VK_EXT_external_memory_host` is not
  enabled and both entry points raise `Feature_not_supported` with the reason. Reason (#503):
  on tartarus (RX 7900 XTX, RADV, Linux 7.1.2-3-cachyos, amdgpu 3.64.0) imports were followed
  by a kernel NULL dereference in `amdgpu_hmm_invalidate_gfx`, hit by `kcompactd` migrating
  the imported anonymous pages, after which processes stayed in uninterruptible sleep until
  reboot. Not reproduced here, on purpose; the cause is in the driver, SPOC's import was the trigger.
  Test: `test_vulkan_host_import_policy` (pure decision table, and on a device the default refusal).
- Vulkan launch batching (#491) is the default again, on every driver including AMD:
  `SPOC_VULKAN_BATCH` defaults to 32; `SPOC_VULKAN_BATCH=1` (submit and wait at every launch) is
  the fallback switch. It had been made opt-in because of #504: on tartarus (RX 7900 XTX, RADV,
  Linux 7.1.2-3-cachyos) `test_vulkan_async` produced GPU write page faults (`gfxhub`,
  0x8001000c0000-0x8001000c3000) and two gfx ring timeouts and resets while reporting success.
  **The cause of #504 is unconfirmed and the issue stays open.** Evidence for restoring: the
  Khronos validation layer (synchronisation validation and GPU-assisted validation, batch 32)
  reported no hazard, and every pair of the async tests (10-15 repetitions each, in parallel) and
  single runs, each under the kernel-log gate, were clean (SPOC 5cc8f0b5, RX 7900 XTX only,
  raw output in the owner comment on #504, 2026-10-12; restored by owner decision, 2026-10-12;
  the attended run of the old stanza that comment suggested was not made). "No failure observed" does
  not separate a fixed state from an unreproduced one, hence the open issue. The import
  hypothesis (#503 userptr work running beside the async tests) is not established: the git
  history does not show the import tests existing at the time of the faults. What changes in the
  tests regardless: the async, event and copy-domain tests set `SPOC_VULKAN_HOST_IMPORT=0` and
  only `test_vulkan_ranged_copies` and `test_vulkan_host_lifecycle` set `1` (their import variants
  skip when import is refused, always on AMD); every Vulkan test that may open a device holds one
  dune lock (`vulkan-gpu`), so no import runs beside an async test and no two of them run at
  once. The batching tests run again on AMD
  drivers (their skip-on-AMD logic is gone) and still set batch 32 themselves. The
  `scripts/gpu-kernel-log-gate.sh` wiring of `make test-vulkan-gpu`, `e2e-gpu` and `e2e-hip` is
  unchanged. If a fault recurs, run with `SPOC_VULKAN_BATCH=1` and report it on #504. The host
  cost of a launch is 6.35 us (3 arguments) / 8.18 us (10) at batch 32 against 52.6 / 55.5 us at
  batch 1 (`bench_vulkan_launch_overhead`, 10000 launches, RX 7900 XTX, #504 comment). The import
  safety text above is unchanged: import stays opt-in and is refused on AMD drivers. The batch size is a tuning
  (family T), not a per-kernel decision, so `SAREK_EXPLAIN` does not report it (as for
  `SPOC_METAL_BATCH`).

### Added

- `Sarek_policy` (`spoc/ir/Sarek_policy.mli`, `specs/optimisation-policy.md`,
  milestone M1): one typed store for every precision grant and consent. A
  `grant` is a `Deviation` (`Accumulation_order`, `Operand_rounding` per
  precision, `Single_rounding`, `Cross_kernel_contraction`) or a
  `Consent Automatic_float_rewrite`; `Sarek_policy.grant scope g ~reason`
  records it for the process or for one kernel (`Kernel "sarek_kern"` and a blank
  reason are refused), `reason ?kernel` reads it (the kernel's own grant first,
  then the process one), and `budget ~kernels` returns the `Budget.t` of a launch
  built from several kernels, the meet of theirs (`Budget` has `exact`, `add`,
  `permits`, `meet`, `leq`, `grants`, `digest`). **No behaviour change.**
  `Sarek_coopmat.accept_relaxed_accumulation`, `accept_operand_rounding`,
  `accept_automatic_float_rewrite` and `Sarek_f16_policy.accept_single_rounding`
  are now wrappers of `grant`, with the same refusals and the same messages;
  `Sarek_coopmat.operand_precision` is an equal type re-exporting
  `Sarek_policy.operand_precision`, and `Sarek_f16_policy.refusal` is
  `Sarek_policy.refusal`. In M1 only the f32 to f16 narrowing reads a
  kernel-scoped grant; a `Kernel`-scoped coopmat grant is recorded and read from
  M2, as is `Cross_kernel_contraction`. No environment variable is added or
  changed. Coverage: no GPU needed. A golden captured on `main` before the change
  (`sarek/tests/codegen_golden/test_policy_golden.expected`) holds the source of
  the CUDA-C (the generator HIP and CUDA/C share), OpenCL, Metal, GLSL, WGSL and
  PTX generators for nine named kernels (a generator that refuses a kernel is
  recorded as its refusal), the coopmat policy verdicts, the grant readers and
  the refusal messages, in five process states; it is byte-identical after the
  change. The laws of `Budget` are checked over all 128 subsets of the seven
  atoms (`spoc/ir/test/test_sarek_policy.ml`). Native and Interpreter read the
  same store through the same `Sarek_coopmat.policy_verdict`; the grants
  themselves are host-side and were not run on a device.
- `scripts/gpu-kernel-log-gate.sh`: runs a GPU command and fails when `journalctl -k` shows
  `page fault`, `ring ... timeout`, `Oops` or `BUG:` since the run started (and refuses to start
  when the last ten minutes already show one). `make test-vulkan-gpu`, `make e2e-gpu` and
  `make e2e-hip` run through it; `make test-gpu-kernel-log-gate` is its red-path harness.

- The float matrix-product rewrite runs on Metal (H4b,
  `specs/matrix-pattern-rewrite.md`, "Fragment shape and Metal"). The rewrite
  templates take the fragment edge (`Sarek_matrix_rewrite.kernel ?frag`,
  `config_of ?frag`; 16 by default with byte-identical output, 8 for Metal's
  `simdgroup_matrix`): a wave still computes a 32x32 tile, as 4x4 fragments of
  8x8, two chained 8-wide multiply-accumulates per step of 16. `Matrix_specialise`
  reads the edge from the device's advertised configurations and admits Metal
  for the float16 operands (f16 x f16 + f32 -> f32) under the unchanged float
  opt-ins (`accept_relaxed_accumulation`; automatically also
  `accept_automatic_float_rewrite`, #440). **The integer rewrite is not ported:
  MSL has no integer `simdgroup_matrix`**, so an integer product runs as the
  original on Metal and the report says why; the bfloat16 lowerings are declined
  too. The advice text lists Metal. A fragment edge other than 16 is admitted only on Metal's f16 operands (the only place it was executed); the device-admission code moved to the new `Matrix_admission` module. The direct float16 size class stays
  16x16-only. Coverage: HIP/Vulkan/CUDA IR byte-identical (golden digests
  unchanged); interpreter at fragment 8 and 16 executed
  (`test_matrix_fragment`: bit-identical to the original kernel and the host
  chain over k, with a positive control); Metal codegen: compiler-output
  (`test_metal_matrix_rewrite`, MSL digest); executed on Apple M4,
  macOS 15 (2026-10-12): `test_matrix_rewrite_metal` (alias `e2e-metal`) passes,
  bit-identical to the original kernel, the interpreter and the host chain, with
  controls rejected; `bench_metal_matrix_rewrite` (9 interleaved pairs, batch 5,
  load 2.6): off 910 / 6802 / 53785 us against on 189 / 966 / 7294 us at
  n = 512 / 1024 / 2048, 4.81x / 7.04x / 7.37x (raw:
  `spoc-pipeline-logs/pkgH/H4b-ladon-bench.txt`).

- Vulkan supports the memory API that HIP, CUDA and Metal had
  (H7 slice 1, `specs/vulkan-async-api.md`): device-only vectors, ranged
  host<->device copies, `copy_bytes_to_host`, `stage_to_device` (synchronous),
  `alloc_pinned`, `register_host`, `event_query`, `stream_wait_event` and
  launches on a stream (`run_source_on_stream`). They were all refused with
  `Feature_not_supported`. Copies complete before they return and are ordered
  against the launches of every stream (`async_copy` stays false; asynchronous
  copies are the next slice). Pinned and registered memory are host ranges
  imported with `VK_EXT_external_memory_host`, so a copy from or to them is one
  DMA `vkCmdCopyBuffer` with no staging buffer: on an RX 7900 XTX (RADV, Mesa
  26.1.4) 64 MiB host-to-device 22 GB/s against 5.6 GB/s through staging, and
  device-to-host 22 GB/s against 2.6 GB/s (provisional, machine load about 4.7).
  A device without the extension refuses both with `Feature_not_supported`.
  Ranged reads never read an uncached BAR mapping with `memcpy` (0.03 GB/s
  measured): they use a HOST_CACHED staging buffer or DMA. Tests: the shared
  device-buffer suite now runs on Vulkan (`test_device_buffers`), and
  `test_vulkan_ranged_copies` (byte-exact on every path with a positive
  control, staging-free assertion, ordering against unsynchronised launches,
  misuse, `event_query`); bench `bench_vulkan_pinned_copy`. The whole-buffer
  `Memory.device_to_host` no longer reads an uncached BAR mapping either
  (0.03 to 2.6 GB/s at 64 MiB). `stream_wait_event` is recorded: synchronising
  the waiting stream also waits for the event. Not yet done: real asynchronous
  copies and a staging pool.

- The int32 matrix-product rewrite runs on Vulkan and is offered on CUDA/C
  (H4, `specs/matrix-pattern-rewrite.md`, "Subgroup width and backends"). The
  rewrite templates take the subgroup width (`Sarek_matrix_rewrite.kernel
  ?subgroup`, `launch_dims ?subgroup`): a block is four waves of the device's
  width, 128 threads on 32-wide HIP and CUDA, 256 on RADV (subgroup 64); the
  HIP output is byte-identical. `Matrix_specialise` admits Vulkan (executed on
  the RX 7900 XTX, bit-identical to the original; integer lowering only) and
  CUDA/C (`Cuda_coopmat` probe: nvcuda::wmma, sm_72+, warp 32) on explicit
  request only (`Sarek_matrix.run`), never automatically: compiler-output
  evidence (NVRTC), never executed; ZLUDA lacks `wmma.*` and unsigned 8-bit
  `mma`, so ZLUDA devices are declined. Because CUDA/C now reports a
  cooperative-matrix probe, the general `SCoopmat` launch gate also admits
  hand-written kernels on CUDA/C (sm_70+ float16, sm_72+ 8-bit, same-sign):
  compiler-output evidence only, never executed on NVIDIA hardware. Vulkan is
  rewritten automatically on 64-wide subgroups only (32-wide on request); the
  rewritten launch is checked against the device's threads-per-block and
  shared-memory limits. The float rewrites stay on HIP and the
  interpreter. Tests: `test_matrix_rewrite` (now enumerating Vulkan),
  `test_matrix_subgroup`, `test_matrix_template_golden`,
  `test_cuda_coopmat_probe`, `test_matrix_cuda_compile`; bench
  `bench_matrix_int_rewrite`.
- `sarek/tests/e2e/test_epilogue_exact.ml` and `bench_epilogue.ml`: epilogue
  fusion against the unfused pair on every deferred device. On HIP (the only
  default backend) fused == unfused is asserted for a forwarding pair, a
  shared-subexpression pair and a chain, on inputs where an fma differs from
  multiply-then-add in 1109 of 4099 elements; on opt-in backends the result is
  recorded, not asserted. Observed, not changed (`docs/fp-contraction-policy.md`
  section 14): OpenCL (rusticl) and Vulkan (RADV) contract a single-kernel
  `a*b+c`; Vulkan's fused kernel differs from the unfused pair when the consumer
  recomputes the producer's product (4099 of 4099).

### Changed

- Every switchable optimisation now decides through `Sarek_policy`, per launch,
  from the effective policy of the kernel being decided (milestone M2 of
  `specs/optimisation-policy.md`): the grants at process and kernel scope (the
  meet over the members of a fused kernel, `Sarek_policy.register_fused`) and
  the process restrictions (`Sarek_policy.restrict`, used by M3's
  `SAREK_OPT_DISABLE` and by tests). New in `Sarek_policy`: `restrict`,
  `restricted`, `register_fused`, `effective`, `effective_for`, `requires`,
  `decide`, `version`, `set_reporter`/`report`. A `Kernel`-scoped grant of any
  of the seven atoms now changes the decision of that kernel and of no other
  (coopmat verdicts and the matrix rewrite read the kernel's budget:
  `Sarek_coopmat.policy_verdict_for`, `Matrix_specialise.lowering_for ~budget`,
  the interpreter's environment carries the kernel's budget). A `Kernel` grant
  on a fused kernel's own generated name is not counted. **Behaviour changes:**
  (1) a grant given after a kernel's source was generated is honoured by the
  next launch (the launch source memo is stamped with the policy; the "a kernel
  generated before the call keeps its barriers" caveat of #480 is gone);
  (2) `Execute.Window.set_epilogue_backend ~backend true` no longer enables
  epilogue fusion on CUDA, OpenCL or Vulkan: it only undoes a switch-off, and
  fusion there needs `Cross_kernel_contraction` for both kernels of the pair,
  with which it now runs (it was opt-in by that call);
  `Launch_epilogue.default_backends` is replaced by `Launch_epilogue.decide_pair`;
  (3) `Sarek_policy.restrict Narrowing_proof`, `Auto_unroll`, `Launch_window`,
  `Matrix_rewrite`, `Epilogue_fusion` switch the optimisation off. HIP and Metal
  fusion, the window, the unroll and the matrix rewrite are unchanged when
  nothing is granted or restricted (goldens unchanged:
  `sarek/tests/codegen_golden/test_policy_golden.ml`). Coverage of epilogue
  fusion by backend (device: RX 7900 XTX, 1109 of 4099 inputs separate fma from
  multiply-then-add; "fused/unfused" = elements of 4099 that differ):

  | Backend | Before | After | Evidence |
  |---|---|---|---|
  | HIP | on, exact | on, exact, no grant | executed: 0 differ from unfused and from the Interpreter, five scenarios |
  | CUDA/PTX | off; `set_epilogue_backend true` enabled it | needs the grant | executed through ZLUDA (not NVIDIA hardware): fused path taken, 0 differ |
  | CUDA/C | as CUDA/PTX | needs the grant | decision tested only (backend not enumerated on the test host, no device run) |
  | OpenCL (rusticl) | as CUDA/PTX | needs the grant | executed: fused path taken, 0 differ (forwarding, shared, chain) |
  | Vulkan (RADV) | as CUDA/PTX | needs the grant | executed: fused path taken; forwarding and chain 0 differ, residual pair 4099 differ by at most 0.5 ulp of the product, within the bound |
  | Metal | on, exact, while `Fp_exactness.is_exact "Metal"` | unchanged; refused if marked inexact even with the grant | executed on an Apple M4 (2026-10-12): `test_epilogue_exact` fused == unfused == Interpreter, 0 of 4099 differ in forwarding, shared and chain; `test_epilogue_metal_pragma` and `test_launch_window` pass |
  | Native, Interpreter, WebGPU | never fuse | never fuse | decision table, exhaustive (`test_sarek_policy_effective`) |

  Raw logs: `spoc-pipeline-logs/policy-impl/m2-*.txt` and, for Metal,
  `spoc-pipeline-logs/policy-impl/ladon/sarek-policy-m2r2-*.txt` (owner machine,
  outside the repository). The bound on a contracting backend is
  `|f - u| <= ulp(a*b)/2 + ulp(r)` (`specs/epilogue-fusion.md` FR-107), not "1
  ulp of the result": under cancellation of `a*b` and `c` it is many ulps of a
  small result. Also: (4) `Execute.Window.epilogue_on backend` now answers "can
  fusion run on this backend under the process grants": false on CUDA, OpenCL
  and Vulkan until `Cross_kernel_contraction` is granted at process scope
  (kernel-scoped grants are decided per pair, `Launch_epilogue.decide_pair`);
  (5) a rewritten matrix kernel (`<name>_sarek_mm`) has the budget of the kernel it
  came from (`Sarek_policy.register_derived`), so a `Kernel`-scoped opt-in rewrites
  that kernel and launches (it was refused by the capability gate); the static
  fusion passes of `Sarek_fusion` register their members like the window and
  epilogue passes; `Sarek_f16_policy.single_rounding_reason` is the budget's
  reason (a fused kernel has one iff every member does).

- Epilogue fusion is on by default on Metal as well as HIP (H9,
  `specs/epilogue-fusion.md` FR-104; it was opt-in). Every Metal kernel, fused
  ones included, starts with `#pragma METAL fp contract(off)`, which on an
  Apple M4 turned 8773 of 8773 contracted `a*b+c` into 0
  (`docs/fp-contraction-policy.md` section 10), so the fused kernel equals the
  unfused pair. It stays off for Metal in a process where the non-fast-math
  compile options cannot be built (`Fp_exactness`, warning logged). Opt out with `SAREK_LAUNCH_EPILOGUE=0` or
  `Execute.Window.set_epilogue_backend ~backend:"Metal" false`.
  `test_epilogue_exact` now asserts fused == unfused AND fused == Interpreter
  bit for bit on every default backend (HIP: executed on an RX 7900 XTX,
  passes), with controls (inputs separate fma from mul-then-add in 1109 of 4099
  elements; the Interpreter and the fused result differ from the fma
  reference); `test_epilogue_metal_pragma` checks on any host that the
  epilogue-fused and the epilogue-then-horizontal MSL carry the pragma and that
  the same check fails on a source without it. Backend coverage
  (epilogue): HIP on and executed; Metal off to on, executed on an Apple M4
  (macOS 15): forwarding, shared and chain pairs give fused == unfused and
  fused == Interpreter (0 of 4099 elements differ), the fused forwarding result
  differs from the fma reference in 1109 of 4099 (control), in-kernel `a*b+c`
  differs from the rounded value in 0 and from fma in 1109. `bench_epilogue`
  on Metal (24 layers, 50 tokens, 15 interleaved pairs, launches per token 48
  to 24, bit-identical): size 128 462.5 us/token median [409.9 to 484.4] off
  against 414.2 [359.6 to 450.8] on (1.12x); size 512 535.9 [459.5 to 552.1]
  against 532.5 [438.7 to 628.0] (1.01x); size 2048 4204.2 [4174.2 to 4325.2]
  against 4218.5 [4171.1 to 4293.1] (1.00x). Neutral to 1.12x, never measurably
  slower beyond noise (7-pair runs at 512 ranged 0.94x to 1.00x with a
  download running alongside); CUDA, OpenCL, Vulkan opt-in
  (contract in kernels); Native and Interpreter never.

- The launch window (deferral and horizontal fusion) is on by default on Metal
  (H9, `specs/launch-window.md` FR-020); it was opt-in. Opt out with
  `SAREK_LAUNCH_WINDOW=0` or `Execute.Window.set_backend_enabled
  ~backend:"Metal" false`. Backend coverage (window): HIP executed (RX 7900
  XTX), OpenCL executed, Vulkan executed, CUDA/PTX executed via ZLUDA on gfx1100
  (not NVIDIA hardware), CUDA/C by construction, Metal off to on, executed on
  an Apple M4 (macOS 15), Native and Interpreter never deferred.
  `bench_launch_window.exe` (`SPOC_BENCH_FRAMEWORK=Metal`; 24 layers x 18
  launches, 50 tokens, 7 interleaved off/on repetitions, a Flash-MoE download
  running alongside, load 6 to 9, so provisional): run 1 22873.4 us/token
  median [22859.8 to 22932.5] off against 22465.4 [22404.8 to 22498.5] on
  (-1.8%); run 2 21326.3 [21296.4 to 22429.4] against 20875.5 [20843.5 to
  21128.1] (-2.1%); launches per token 432 to 312; host enqueue 4.7 to 5.1 us
  per launch off, 5.2 to 5.7 on; results bit-identical on and off. Metal's 31
  `[[buffer]]` slots split the four-member `independent` scenario of
  `test_launch_window` into 2 launches (the test derives the expected count
  from the window's own slot accounting).

- Metal `waitUntilCompleted`, the two compiles (`newLibraryWithSource`,
  `newComputePipelineState`) and the libc `memcpy` behind host/device copies
  release the OCaml runtime lock (R-24, audit items 16/17), as the other
  backends did: `Metal_bindings` gains a second typed
  `objc_msgSend` stub (`msg_void_blocking_lazy`,
  `~release_runtime_lock:true`) used only for blocking messages, so a domain
  waiting on a command buffer no longer stops every other domain's minor
  collections. Both arguments are C pointers (receiver, selector), no OCaml
  heap pointer is passed, and the caller keeps the command buffer retained
  across the wait (`Metal_api.Pending` and `Event.synchronize` do). Other
  messages stay on the plain stub. Tests: `test_metal_wait_binding`
  (structural, runs anywhere; failed before the change; also checks the compiles and `memcpy`) and the shared
  `test_blocking_ffi_releases_lock` on alias `e2e-metal`. Bench `bench_metal_compile` (alias `e2e-metal-bench`): library
  and pipeline-state compile time of three decode-shaped kernels, cold vs warm,
  each in a new process. Measured on Apple M4 (macOS 15, 5 rounds, kernel order rotated, nonce in the
  compiled code), a source new to the machine costs about 7.0-12.2 ms per kernel,
  library plus pipeline (matvec 3.6 + 6.0, RMS norm 3.4 + 3.6, RoPE 2.6 + 9.5 ms) and
  a new process after that at most 0.04 ms: the OS shader cache already serves
  a new process, so Metal gets no disk compile cache of ours (not needed;
  `MTLBinaryArchive` could win nothing measurable; raw:
  `docs/measurements/metal-runtime-lock-2026-10-12/compile-bench.txt`).

  | Backend | Before | After | Evidence |
  |---|---|---|---|
  | HIP, CUDA, OpenCL, Vulkan | lock released (R-24) | unchanged | executed (RX 7900 XTX; CUDA via ZLUDA) |
  | Metal | lock held during `waitUntilCompleted` | released | structural test on tartarus; behaviour executed on Apple M4, macOS 15 (ladon, 2026-10-12): the other domain completed 1000 minor GCs during a 199 ms wait, against 1 on origin/main 1808a9a5 (raw: `docs/measurements/metal-runtime-lock-2026-10-12/r24-ladon.txt`) |
  | Native, Interpreter | no blocking driver call | unchanged | by-construction |

- CUDA kernel launches marshal all arguments into one 8-byte-slot block plus one
  pointer array (two host allocations whatever the argument count), as the HIP
  launch already did (`specs/cheap-repeated-launches.md` US-4, C-13). Both
  blocks are retained until the stream or device is retired, as before. Each
  value is written with its own C type at its slot start. Host cost per launch
  of an empty PTX kernel, executed via ZLUDA on an RX 7900 XTX (not NVIDIA
  hardware), median of 7 interleaved pairs at load ~9 (provisional): 3 arguments
  2.81 us to 2.21 us, 10 arguments 4.85 us to 2.56 us
  (`sarek-cuda/test/bench_cuda_kernargs.exe`). Test: `test_cuda_kernargs`.

### Fixed

- CUDA/PTX kernels load on Blackwell (sm_120, #506). `load_from_ptx` rewrote
  `.target` to the device SM but kept the generated `.version 8.0`, which the
  driver rejects for sm_120 (needs PTX ISA 8.7). The rewrite now raises
  `.version` to the ISA the device SM needs (never lowers it); the SM-to-ISA
  table moved from the test tree to `Cuda_ptx_isa`. A driver whose CUDA version
  is known to accept a lower ISA fails with a message naming both versions
  instead of `CUDA_ERROR_INVALID_PTX`. The NVRTC path is unchanged (it compiles
  for at most compute_90 and the driver JITs it). Not run on Blackwell hardware.
- The launch window no longer grows a fused group past Vulkan's 128-byte
  push-constant block (H6). A fused kernel carries every member's vector
  lengths and scalars plus an int32 offset and size per member, so six small
  launches of one vector and three int32 scalars (24 bytes each) made a
  144-byte block that the backend refused: the window logged the failure,
  issued the six launches one by one and never fused that sequence again. On
  SPOC e9f22fe9 the same refusal also left the stream fence unsignalled and
  the next launch hung in `vkWaitForFences` (fixed earlier by a312b91f and
  63f5f0e9; the fence is now reset only just before `vkQueueSubmit` and tracked
  as pending only once submitted). The window now sizes the
  block before it grows a group or merges an epilogue, with
  `Sarek_push_constants` (the layout function `build_push_constants` also
  writes with) against `Launch_window.push_constant_limit` (Vulkan: 128). A
  single launch over the limit raises a "push constant block overflow" error.
  Spilling scalars to a buffer, and reading `maxPushConstantsSize`, are not
  done. Tests: `test_vulkan_window_push_budget` (Vulkan, RADV NAVI31),
  `test_vulkan_push_constant_budget`, `test_sarek_push_constants`,
  `test_launch_window`.

- CUDA/PTX emits `global_idx_z` (H8). The PTX intrinsic table had
  `global_idx_x` and `global_idx_y` but not `global_idx_z`, so any 3-D kernel
  using it raised `Ptx_codegen_error: unsupported construct: intrinsic:
  global_idx_z` while CUDA C, HIP, OpenCL, GLSL, WGSL and Metal accepted it. It
  is `%tid.z + %ctaid.z * %ntid.z` in u32 registers, like x and y. An audit of
  `Sarek_ir_intrinsic_dispatch.thread_intrinsic_names` found no other PTX gap.
  Test: `test_ptx_global_idx_z` (every name lowers; a 3-D launch equals the
  Interpreter, executed via ZLUDA on gfx1100, not on NVIDIA hardware).
  `test_matrix_rewrite` now runs the corpus on CUDA/PTX too (the H4 skip is
  removed).
- A `for` upper bound is evaluated once on every backend (compiler review
  C-8). CUDA/HIP, OpenCL, Metal, GLSL and WGSL printed
  `for (i = a; i <= bound; i++)`, which re-reads the bound each iteration, so
  `let m = mut 3l in for j = 0 to m do m := m - 1l ... done` ran 2 times on a
  GPU and 4 on the interpreter, Native and PTX. A bound the body can change (an
  assigned variable, an array element, a call) is now bound to a
  `sarek_for_hi_N` temporary before the loop (`Sarek_ir_for_bound`); a bound the
  body cannot change, such as `n - 1l`, keeps its generated source. A bound of
  `Int32.max_int` (`to`) or `Int32.min_int` (`downto`) remains unsupported on
  every backend (the loop variable overflows). Tests:
  `test_hip_cfamily_semantics` (HIP against the interpreter),
  `test_for_bound_signed_idx` (compile-only goldens).

- Thread, block and grid indices are signed `int` on CUDA/HIP, OpenCL and Metal
  (compiler review C-9). `threadIdx.x`, `get_local_id(0)` and Metal's `uint3`
  members are unsigned, so an inline `thread_idx_x - 1l < 0l` was always false
  and `(thread_idx_x - 5l) / 2l` gave 2147483645 instead of -2; they are now
  spelled `((int)threadIdx.x)` etc., as GLSL and WGSL already were. The
  generated source of every kernel that reads a thread index changes (casts
  only). Metal's `global_size` is also parenthesised now.

- The divergent-barrier check works under `let open Sarek_stdlib.Std in`
  (#447). The PPX auto-opens `Gpu`; an explicit `let open Std in` then replaced
  each `Gpu` entry with the registry's, which carries no convergence class, so
  `block_barrier` (and the other convergence points) lost the requirement and
  `if thread_idx_x > 16l then block_barrier ()` compiled in the documented
  style. The recorded class is now kept. A kernel that was accepted only
  because of this and has a barrier under thread-varying control is now
  rejected, as it is without the open (`sarek/tests/negative/test_shared_tuple.ml`
  was one). The qualified spelling `Gpu.block_barrier ()` had the same hole
  (registry entries carry no convergence class) and is fixed at the source.
  The native backend's barrier detection reads the same class: a kernel with
  `block_barrier` under `let open Std` or written `Gpu.block_barrier` was
  compiled as barrier-free (barrier a no-op, parallel pool), and now gets the
  barrier-aware path. Regression tests:
  `sarek/tests/negative/test_barrier_open_std.ml`,
  `sarek/tests/negative/test_barrier_qualified_gpu.ml`,
  `sarek/tests/unit/test_convergence.ml` ("name resolution").

- The fallback of an epilogue-merged launch (the fused kernel failing at issue
  time) now issues the pair like the launch window does (#459). If the
  producer raised, the consumer was never issued, against the window's
  contract that every held launch is issued and the first exception re-raised
  afterwards; and both launches were counted as one in `issued`. Regression
  test: `sarek/tests/unit/test_launch_epilogue_fallback.ml`.

- `SAREK_DEBUG=kernel` (and any other component named without a level) now
  prints that component's debug messages, as the backend READMEs say (R-43).
  Before, the level stayed at `warn`, so `Log.debug` printed nothing. A level
  in the list still wins: `SAREK_DEBUG=kernel,info` prints info and above.
  Linking `Spoc_core.Error` no longer enables every `Log` component at load
  time; `Error`'s own messages are gated by the level only
  (`Log.log_any_component`). A program that links `Error` and relied on that
  to see `Log.warn`/`Log.error` from components it never enabled must now
  name them in `SAREK_DEBUG` or call `Log.enable`. Regression tests:
  `sarek/core/test/test_log.ml`, `sarek/core/test/test_log_error.ml`.
- `Device.synchronize` now releases the stream keep-alive ledger of the
  device, as `Transfer.flush` does (R-6). Before, after a `Device.synchronize`
  `free_pinned`, `unregister_host` and `free_buffer` still refused with "still
  referenced by stream-ordered work", and a program that only used
  `Device.synchronize` kept every stream operand rooted for the life of the
  process. `Device.on_synchronize` registers a hook run around each
  whole-device sync; `Transfer` marks each stream's last operation before the
  sync and releases up to those marks after it, so work another domain
  enqueues during the sync stays rooted. `Transfer.flush` is now
  `Device.synchronize`. On OpenCL, `Device.synchronize` now finishes every live
  stream's command queue, not only the default queue, and a device's state
  (context, default queue, stream list) is created once even when several
  domains use the device for the first time at once. Regression tests:
  `test_device_synchronize_releases` and
  `test_device_synchronize_keeps_later_work` in
  `sarek/tests/e2e/device_buffers_suite.ml` (CPU backends and HIP),
  `sarek-opencl/test/test_opencl_device_sync_streams.ml`,
  `sarek-opencl/test/test_opencl_device_state_race.ml`.

- Epilogue fusion no longer accepts a producer that stores below index 0 as
  covering its consumer: the fused consumer would then run on negative
  indices the unfused one never touches. Such pairs stay unfused. A refusal
  found while enumerating (an element stored twice, an index out of range, a
  value-dependent unknown) or over the enumeration bound is cached for those
  scalar values only; it used to be reused for every value of the scalars.

- Native launches from several domains no longer hang or corrupt each other
  (R-64). The process-wide `ThreadPool` and `ParallelPool` serialise whole
  launches behind a per-pool `launch_lock`, so counters, the work function,
  `pending_workers` and `first_error` belong to one launch at a time;
  `ParallelPool` completion now wakes all waiters (`Condition.broadcast`; `ThreadPool` already did). A launch on a pool from inside one of its own kernels raises `Invalid_argument` instead of deadlocking; `flush_fission` flushes every queue before re-raising the first error. A kernel that raises on
  a fission queue no longer kills the dispatcher domain: the error is re-raised
  by the next `flush_fission_queue` (R-67). Regression test:
  `sarek/tests/unit/test_cpu_runtime_multidomain.ml`.
- Superstep convergence rule settled (`specs/convergence-semantics.md` §2,
  global review D-38, D-57, C-7). A `let%superstep` reached under
  thread-varying control is rejected with or without `[@divergent]`; the
  attribute is still accepted but no longer has any effect. Before, it skipped
  the check, so `if thread_idx_x > 16l then let%superstep[@divergent] s = ... in ()`
  compiled to a barrier inside a divergent branch. A thread-varying `if`
  inside a superstep body, with no barrier in its branches, stays legal: it
  reconverges before the implicit barrier. The Rocq model
  (`formal/convergence-safety`) implements the same rule; `[@divergent]`
  supersteps are now inside the verified fragment, and the warp checker
  handles varying early returns like the barrier checker (D-56). The
  non-blocking `KNOWN-ISSUE` line of `make test_negative` is gone.
- Formal: the PTX model's right shifts now match the emitter (global review
  D-54). `Shr` was modelled as a logical shift with no `Lshr32`/`Lshr64`;
  production emits `shr.s32`/`shr.s64` for `Shr` and `shr.b32`/`shr.b64` for
  `Lshr`. `formal/codegen-ptx/theories/PtxShift.v` models arithmetic and logical
  shifts over two's-complement patterns with PTX count clamping (18 theorems);
  the model is extracted (`ShiftExtract.v`) and the conformance test checks it
  against `Sarek_ir_ptx_expr` (instruction chosen, and the result of the
  emitted instruction on negative operands and counts beyond the width) instead
  of against a hand copy of itself. Swapping `s`/`b` in production or in the
  model turns the test red.

- Formal gate: every committed extracted model is now drift-checked (global
  review D-55). Previously only the PTX layout model was compared with a fresh
  extraction; the twelve type-safety models and the convergence model, which
  the conformance suites use as oracles, were not, and the type-safety
  extraction drivers were not in `_CoqProject`. `scripts/check-formal-proofs.sh`
  now regenerates and compares all of them (15 models) and fails on a driver
  missing from `_CoqProject`, an orphaned model, or an unnamed exemption. The
  committed `TypeSafetyModel.ml` had gone stale (no `TRecord`/`TVariant`) and was
  regenerated. Committed models are stored in canonical form
  (`scripts/canonicalize-extraction.py`) and listed in `.ocamlformat-ignore`.

### Added

- OpenCL keeps compiled programs on disk (package H, H5c). `Program.build`
  stores the driver's program binary (`clGetProgramInfo`) through
  `Framework_cache`, the store Vulkan already used for SPIR-V, and a later
  process rebuilds from it with `clCreateProgramWithBinary`. The key holds the
  platform name and version, device name and version, driver version, the exact
  build-option string and the source. A binary that is damaged, refused by the
  driver or fails to build is replaced by a source build and the entry is
  rewritten; a driver that returns no binary is not cached. Measured on
  tartarus (provisional, load > 4): build of three decode-shaped kernels in a
  new process, Intel OpenCL CPU 91 / 33 / 42 ms -> 1.6 / 1.1 / 1.2 ms; rusticl
  (RX 7900 XTX) 7.7 / 3.8 / 5.3 ms -> 2.2 / 1.9 / 4.2 ms with Mesa's shader
  cache off, 1.8 / 0.8 / 0.9 -> 0.7 / 0.5 / 0.6 ms with it on. Output bits are
  identical to a source build. `SAREK_DISK_CACHE=0` switches the disk cache off
  for OpenCL and Vulkan (new in `Framework_cache`, with
  `compute_key_with_options`); `Framework_cache.put` now writes a temporary
  file with a unique name and renames it, so a reader (any thread, domain or process) sees one complete entry and never a half-written one. The cache is refused when its directory is not owned by the user, is group- or other-writable, or would be the shared temporary directory.
  Tests: `sarek-opencl/test/test_opencl_disk_cache.ml`,
  `sarek/framework/test/test_framework_cache.ml`; bench:
  `sarek-opencl/bench/run_ab.sh`. Docs: `gh-pages/docs/environment.md`.

- `SCoopmat` and scalar `float16` on Metal (package H, H3). Metal lowers
  cooperative matrices to `simdgroup_matrix` (8x8x8, `f16 x f16 + f32 -> f32`,
  both memory layouts, one SIMD-group of 32 per fragment); the Metal plugin now
  probes the configuration by compiling it on the device and reads the SIMD
  width from the pipeline (`coopmat = Some ...`, `warp_size` from the device).
  Scalar `f16` is lowered to `half` (the narrowing is measured strict on the
  M4). Everything else is refused with its reason at code generation: integer
  components (MSL has no integer `simdgroup_matrix`), f16 accumulation (no
  closed form), bf16, f32 operands, other shapes, a block that is not whole
  32-wide SIMD-groups. Like HIP, the float configuration needs
  `Sarek_coopmat.accept_relaxed_accumulation`; on Apple M4 (the only Metal device run) it is
  bit-identical to the interpreter (accumulator = C, products added in index order in binary32).
  `Sarek_gemm.Coopmat` takes `?frag` (16 by default, 8 on Metal). OpenCL still
  has no lowering and its refusal now says why: no reachable device advertises a
  matrix extension. Spec `specs/coopmat-metal-simdgroup.md`; tests
  `test_metal_coopmat_golden`, `test_metal_coopmat_probe`,
  `test_coopmat_metal_e2e` (Apple GPU), bench `bench_metal_coopmat_gemm`.
  Executed on Apple M4 (ladon, macOS 15, 2026-10-12): `test_coopmat_metal_e2e`
  PASS (all controls rejected) and `test_metal_f16_narrowing_e2e` PASS (0/63488
  against the interpreter and the model on both shapes); `test_metal_f16_narrowing_e2e` runs the Sarek-generated
  scalar f16 kernel over all 63488 finite binary16 values (both swept shapes,
  bit-equal to the interpreter and an independent model, with controls).

- `Spoc_core.Advanced.Pinned` is real (global review D-69). It was a placeholder
  whose `create` returned an ordinary pageable vector and whose `is_available`
  was hard-coded `false`. It is now an explicit, opt-in owner handle over
  `Transfer.alloc_pinned` / `register_host`: `alloc`, `register`, `bigarray`,
  `release`, `with_alloc`, `is_live`, `device`, and an `is_available` that
  probes the backend. **Breaking:** `Advanced.Pinned.create` is removed (no
  caller in the tree); it could not be honest, since a vector's host mirror is
  garbage collected and pinned memory is released only explicitly.
  The handle type is abstract; `with_alloc ?stream` synchronises the stream
  before releasing. `Advanced.has_advanced_features` now answers from the
  `Pinned.is_available` probe (a one-byte allocation) instead of always
  `false`. `Advanced.Unified` stays a placeholder. Test:
  `sarek/tests/e2e/test_advanced_pinned.ml`; docs: `memory.md`.

- `Sarek_f16_policy.accept_single_rounding (Kernel name | All_kernels)
  ~reason : (unit, refusal) result` (refuses a blank reason and
  `Kernel "sarek_kern"`, the name of every unnamed kernel): an explicit opt-in that removes every f32 -> f16 narrowing barrier
  from the covered kernels (global review C-40, specs/f16-narrowing-barrier.md
  §4). The device may then round a narrowed expression once where the
  interpreter rounds twice: at most one binary16 ulp plus `|a * b| * 2^-24` for
  a fused `a * b +/- c`, and a `-0` product may come out `+0`. Separate from
  `Sarek_coopmat.accept_relaxed_accumulation`. On a dequantise-to-LDS
  microbenchmark shaped like Katra's fused Q4_K (hip:0), 870 -> 170 us.
  Differential tests of a covered kernel must compare within that bound
  (`sarek-hip/test/test_hip_f16_single_rounding.ml`).

- `Spoc_core.Host_ptr_buffer`: an explicit, typed opt-in to give a vector a
  CPU OpenCL buffer over its host memory (`CL_MEM_USE_HOST_PTR`), for large
  host pools that must not be duplicated. `of_bigarray dev kind ba` and
  `attach vec dev` return a `result`; refusals are structured
  (`Framework_sig.host_ptr_refusal`: not a CPU device, unsupported backend,
  empty, misaligned against `CL_DEVICE_MEM_BASE_ADDR_ALIGN`, larger than
  `CL_DEVICE_MAX_MEM_ALLOC_SIZE`; or a vector that already has a buffer on the
  device). `publish_host_writes` and `acquire_device_writes` state the
  map/unmap synchronisation; `host_link vec dev` reports `Separate`,
  `Host_storage` or `Host_ptr`. Backends implement the new
  `Framework_sig` `Memory.alloc_host_ptr`, and `Memory.BUFFER` has a new
  `host_link` value. Measured on the Intel CPU runtime: a kernel reading a
  3 GiB array grew VmRSS by 116.6 MiB through a host-pointer buffer and by
  3188.7 MiB through the copy path (`docs/measurements/opencl-host-ptr-2026-10-11/`).
  Guide: `gh-pages/docs/memory.md`, "Sharing host memory with a CPU OpenCL
  device".

- Cooperative-matrix loads and stores take a memory layout (C-41):
  `CM_load` / `CM_store` have a required field
  `layout : Sarek_coopmat_types.layout` (`Row_major | Col_major`, re-exported
  by `Sarek_coopmat`); column-major puts element (r, c) at
  `index + c * stride + r`. Lowered by the interpreter, HIP gfx11 (a
  column-major B fragment is the contiguous helper: two `global_load_b128`
  per lane instead of 16 `global_load_d16`), CUDA (`col_major` fragment
  types, `mem_col_major`; compile-only) and GLSL
  (`gl_CooperativeMatrixLayoutColumnMajor`); refused where cooperative
  matrices are. `Sarek_gemm.Coopmat.ir` / `run` take `?b_layout` (B stored
  transposed, `[n][k]`). Row-major kernels generate the same kernel bodies as
  before (the shared helper preamble changed). **Migration:** clients
  constructing `CM_load` / `CM_store` record literals add
  `layout = Row_major` to each (the field has no default, so each construction
  site states its layout). Tests: `sarek/tests/unit/test_coopmat_layout.ml`,
  `sarek/tests/codegen_golden/test_coopmat_layout_codegen.ml`,
  `sarek/tests/e2e/test_coopmat_layout_e2e.ml` (HIP gfx1100 and Vulkan RADV).

- Matrix products are rewritten at launch, inside `Execute.run_vectors` (spec
  `specs/matrix-pattern-dynamic.md`). A kernel that is a recognised matrix
  product runs on matrix units when the device has them (HIP gfx11; the
  interpreter only on explicit request, so that it stays the oracle): int32
  by default (bit-identical), float only after
  `accept_automatic_float_rewrite` and the accuracy opt-ins (see Changed).
  `SAREK_MATRIX_REWRITE` (`off`, `exact`, `auto`; accepted spellings in Changed)
  or `Matrix_specialise.set_mode`
  switches it; `Execute.run_vectors_outcome` returns the decision,
  `Execute.run_vectors_plain` launches the kernel as given,
  `Matrix_specialise.set_reporter` / `SAREK_MATRIX_REPORT=1` report it. The
  aligned float16 size class loads fragments straight from global memory
  (`Sarek_matrix_direct`): 256×3072×3072 runs at 39.6–41.9 TFLOP/s on an
  RX 7900 XTX against 34.8–35.2 for the hand-written `Sarek_gemm.Coopmat`
  (three sessions, `bench_matrix_rewrite.exe 3 10 10`, 2026-10-11, spec
  Results). Plans and rewritten kernels
  are cached per IR; a kernel that is not a matrix product costs one memo
  lookup per launch (host launch cost unchanged within run-to-run spread).

- Automatic rewrite of recognised matrix products onto matrix units (spec
  `specs/matrix-pattern-rewrite.md`). `Sarek_matrix.run` (library
  `sarek.matrix`) is a drop-in for `Execute.run_vectors`: when the kernel is a
  GEMM nest whose indices and guards it can bound, the device offers the
  configuration (HIP gfx11, interpreter) and the policy admits it, it launches a
  generated SCoopmat kernel (64×64 tiles staged through shared memory, any size,
  layout, batch, epilogue) over exactly the outputs the original launch writes,
  and otherwise the original; the outcome says which and why. int32 products
  are rewritten by default through 8-bit byte planes and are bit-identical;
  float16-operand products need `accept_relaxed_accumulation`; float32 products
  need `~precision` and the new `Sarek_coopmat.accept_operand_rounding`
  (`Operands_f16`, `Operands_bf16`, `Operands_bf16x3`: a three-term bfloat16
  split, 0.04–0.15 γ_K measured). New: the `BFloat16` cooperative-matrix
  component (bf16 operand loaded from a float32 buffer with rounding; HIP gfx11
  and interpreter), `ECast (TUint8, _)` as a narrowing store, host evaluation of
  uniform expressions (`Sarek_ir_eval_int`). On an RX 7900 XTX the rewrite is
  1.4–2.5× (int32) and 3.4–9× (f16) faster than the naive kernel, and 4–6×
  slower than the hand-written `Sarek_gemm.Coopmat`.

- Epilogue fusion in the launch window (spec `specs/epilogue-fusion.md`). A
  pending launch P and an element-wise consumer C that reads P's output at its
  own index (`let i = global_thread_id in ...`) are merged: C's body runs right
  after each of P's stores, in the thread that made the store.
  - The merge happens only when the host evaluation of both launches' index
    computations (`Index_slice`) shows that P stores every element C writes,
    exactly once. C must also read nothing else P writes, and write nothing P
    accesses except the intermediate.
  - Chains and horizontal groups compose.
  - On by default on HIP only, because its compiler is forced to
    `-ffp-contract=off`. Elsewhere a compiler could contract across the
    merged code, so it is opt-in with `Execute.Window.set_epilogue_backend`.
    `SAREK_LAUNCH_EPILOGUE=0` turns it off.
  - spoc_llm decode, RX 7900 XTX: 335 → 281 launches/token on the 0.8B and
    749 → 665 on the 35B, logits bit-identical.

- Launch window with horizontal fusion (spec `specs/launch-window.md`).
  `Execute.run`/`run_vectors` JIT launches without `~stream` are deferred on
  HIP, OpenCL and Vulkan up to the next synchronisation point:
  - every Transfer read-back, upload into a buffer a pending launch binds,
    copy, staging, free, flush, stream and event operation;
  - `Device.synchronize`, `Profiling` events and timers;
  - raw `Kernel` launches, `run_source`, stream launches and `Soa_launch`;
  - device destroy and cache clear.

  Pending launches that bind no common buffer with a write, and have the same
  block shape and grid y/z, are issued as one kernel. Each work-group runs one
  member's unchanged body, with `block_idx_x`, `grid_dim_x` and the global
  index intrinsics rewritten to the member's own values. The result is
  bit-identical to the unfused launches by construction, on every backend.
  Fused kernels are cached by member sequence and reused across launches.

  Errors move only at device compilation and driver launch time: checks and
  code generation still raise in the launching call, but a failure at those
  two stages is raised by the call that flushed the window. If a fused kernel
  fails, its members are issued one by one.

  Controls: `Execute.Window.set_enabled`, `set_backend_enabled`, `flush` and
  `stats`; `SAREK_LAUNCH_WINDOW=0` turns the window off. These launches stay
  immediate:
  - CUDA, Metal and WebGPU (each can opt in);
  - CPU devices, whose zero-copy buffers alias;
  - Native and Interpreter.

  A kernel is never fused if it reads `grid_dim_x`, uses intrinsics from
  outside `Sarek_stdlib`, or declares shared memory of non-constant size.

  `bench_launch_window` on an RX 7900 XTX runs a decode-like chain of 24
  layers × 18 launches. Window off vs on, 5 interleaved repetitions, median
  µs/token: 4173 → 3158, with 432 → 288 launches per token, and results
  bit-identical. Tested on HIP (RX 7900 XTX, 7950X iGPU), OpenCL (rusticl,
  both GPUs), Vulkan (RADV), Native and Interpreter.
- Matrix-product (GEMM) recognition and a compile-time advice (spec
  `specs/matrix-pattern-advice.md`). `Sarek_matrix_pattern.analyze` finds, in
  a kernel's IR, every loop that accumulates `A[ia] * B[ib]` (or `fma`) over a
  reduction index and stores the sum once to `C[ic]`, with affine indices
  (`Sarek_ir_affine`) that separate into row, column and batch indices:
  row- and column-major operands, 1-D (`tid / n`, `tid mod n`) and 2-D thread
  mappings, loop nests, batches, padded leading dimensions, offsets, an
  accumulator starting from C, and `alpha·acc + beta·C` epilogues; every other
  reduction loop (overlapping or non-unit-stride operands, dependent indices,
  non-affine or gathered indices, ...) is reported with the reason it is not
  one.
  `Sarek_matrix_advice` names the matrix-unit configuration that would apply
  (f16 operands: `f16×f16+f32→f32`, bounded, needs
  `accept_relaxed_accumulation`; float32: needs operand rounding, no opt-in yet;
  int32: exact through 8-bit byte planes) and the backends that run it. The
  PPX reports each such nest as the OCaml alert `sarek_matrix` located at its
  `for` loop: non-fatal, silenced with `[@alert "-sarek_matrix"]` on the kernel
  or `[@@alert "-sarek_matrix"]` on its binding (or per file/library, or made an
  error with `-alert @sarek_matrix`). `SAREK_MATRIX_SCAN=<file>` makes the PPX
  log every finding (fresh build directory, `DUNE_CACHE=disabled`). Nothing is rewritten. `Sarek_fusion.should_fuse` now gives
  a matrix-product operand its own reason. `Sarek_ir_conv` moved from
  `sarek.transpile` to `sarek.frontend` (the PPX converts its IR for the
  analysis). Scanned over the 254 kernels the repository builds (outside the new corpus): 4 advised,
  all matrix products; no false positive.

- `Transfer.copy_bytes_to_host ?stream ~src ~src_byte_offset dst`: the
  byte-level read-back, mirror of `copy_bytes_to_device`. Copies
  `Bigarray.Array1.size_in_bytes dst` bytes of a vector of any scalar element
  kind, from byte `src_byte_offset`, into a C-layout Bigarray of any element
  kind (e.g. a range of an `int32` device-only vector into a `char` slot of a
  host pool). Same stream, keep-alive and location contract as
  `copy_to_host`, with which it shares one implementation (`d2h_range`); the
  source's location is unchanged. With `~stream`, the destination's contents
  are undefined until the stream or a later event has completed, and
  `free_pinned`/`unregister_host` refuse it meanwhile. Refused with
  `Feature_not_supported` on OpenCL, Vulkan and WebGPU. Tested on Native,
  Interpreter, HIP (RX 7900 XTX and the 7950X iGPU) and CUDA through ZLUDA;
  Metal shares the `copy_to_host` path and was not run. Spec
  `specs/copy-bytes-to-host.md`.

- Cooperative matrix (`SCoopmat`) on HIP through the RDNA3 matrix units, and
  an explicit opt-in for float accumulation (spec `specs/coopmat-hip-wmma.md`).
  On gfx11 devices with 32-wide wavefronts the CUDA-C generator (shared by HIP)
  lowers 16×16×16 `f16 × f16 + f32 → f32` to
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` and the eight `u8/s8 × u8/s8 +
  s32/u32` configurations to `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`; the
  HIP backend now reports these through `capabilities.coopmat`, and any other
  HIP device (the gfx1036 iGPU, gfx10, gfx12, CDNA, wave64) is reported with no
  configuration, so the launch gate refuses with the HIP rule in the message.
  Float accumulation is bounded, not bit-identical (Regime B of
  `docs/design/f16-relaxed-accuracy.md`): the launch gate and the interpreter
  refuse it until the program calls
  `Sarek_coopmat.accept_relaxed_accumulation ~reason`; with it, the interpreter
  evaluates the f16 configuration in a fixed sequential binary32 order.
  `Sarek_gemm.Coopmat` is a tiled f16 GEMM built from these statements
  (`d = a·b + c`, row-major). Measured on an RX 7900 XTX (ROCm 7.2.4): the
  integer configurations are bit-identical to the interpreter over all 65536
  8-bit operand pairs; the f16 GEMM is within γ_K·(Σ|a·b| + |c|) of the exact
  result at 64×64×128 and 256×256×3072 (worst 0.044 of the bound), repeatable,
  and identical across tilings; at M = 256, K = 3072, N = 1024/3072/12288 it
  runs at 20–24, 36–43 and 51–53 TFLOP/s (best tiling, median of two sessions)
  against 3.2–3.3 for the scalar f32 `Sarek_gemm`, 10–16 for rocBLAS with f32
  output and 26–84 for rocBLAS with f16 output
  (`sarek-hip/test/bench_hip_coopmat_gemm.exe`,
  `tools/probes/rocblas_gemm_f16.cpp`). The CUDA arm (`nvcuda::wmma`) is
  compile-checked only — NVRTC, and HMMA/IMMA in sm_80 SASS — and stays
  refused at launch because the CUDA backend does not probe `coopmat`. A
  cooperative-matrix launch whose block is not a whole number of 32-wide waves
  is refused. Metal, OpenCL, WGSL and PTX still refuse `SCoopmat`.
  `Sarek_coopmat.device_support` gains `ds_evidence`; `TUint8` is
  `unsigned char` in CUDA-C.

- `Transfer.stage_to_device ?stream src ~dst ~dst_offset`: the cheap way to
  update a small vector that kernels read (at most
  `Transfer.max_staged_bytes` = 64 KiB per call). The source is copied into a
  1 MiB per-device pinned staging ring before the call returns, so the caller
  may reuse it at once, and the copy is enqueued on the stream (default: the
  device's default stream) without waiting for the GPU. `Vector.set` on such a
  vector downloads it after every launch and re-uploads it synchronously at
  the next one, waiting for the GPU queue both times; its documentation now
  says so. Measured with `sarek-hip/test/bench_hip_small_writes.exe` (RX 7900
  XTX, 8 int32 + 8 float32 rewritten, 300 us host work, 4 launches per
  iteration, three runs, 2026-10-10): 705.2–715.1 us per iteration with
  `Vector.set`, 330.1–338.9 with `stage_to_device`, 322.0–322.5 without any
  rewrite. Native, Interpreter and Metal perform a synchronous ranged copy; OpenCL, Vulkan and
  WebGPU refuse. Spec `specs/transfer-pitfalls.md`.

- `Transfer.register_host` documents measured ROCm pitfalls: copies right
  after registering are intermittently slow (7 of 50 first passes at
  0.95–6.17 GB/s, about 26 otherwise, whether the memory was touched before
  registering or not; `bench_hip_register_host.exe`), a copy may not span two
  registrations (`hipErrorInvalidValue`), and ROCm does not count registered
  pages as pinned (`VmPin: 0 kB`).

- Metal implements device-only vectors, ranged copies, pinned memory, streams
  and events (spec `specs/metal-ranged-copies.md`). Buffers stay
  shared-storage MTLBuffers (Apple silicon unified memory), so a ranged copy is
  one bounds-checked `memcpy` at `contents + offset`; `alloc_pinned` is
  page-aligned host memory and `register_host` a no-op (nothing to page-lock
  for the GPU). Buffers above the device's `maxBufferLength` are refused with
  an error naming the limit, and `capabilities.total_global_mem` reports
  `recommendedMaxWorkingSetSize` instead of a 4 GiB constant. Measured on an
  Apple M4 (macOS 15.6.1), 256 MiB ranged copies, best of 5: 42.6 GB/s
  host->device and 42.5–42.7 GB/s device->host, pageable and page-aligned alike
  (`sarek-metal/test/bench_metal_copy.exe`, information only).

- `Int32.float_of_bits : int32 -> float32` and
  `Int32.bits_of_float : float32 -> int32` in the Sarek stdlib: bit
  reinterpretation with OCaml's `Stdlib.Int32` names and semantics, lowered on
  every backend — CUDA C/HIP `__int_as_float`/`__float_as_int`, OpenCL
  `as_float`/`as_int`, Metal `as_type<float>`/`as_type<int>`, GLSL
  `intBitsToFloat`/`floatBitsToInt`, WGSL `bitcast<f32>`/`bitcast<i32>`, PTX
  `mov.b32`; Native and Interpreter run OCaml's own functions (a signalling
  NaN is quieted there, as in OCaml, because float32 is carried as a double).
  The GLSL, WGSL and PTX lowerings only apply to the stdlib `Int32` path, so
  another module's `float_of_bits` is not captured. The C-family backends now print
  the literal `Int32.min_int` as `(-2147483647 - 1)` and WGSL as
  `(-2147483647i - 1i)`: `-2147483648` is a `long` in C/OpenCL/MSL, which
  `as_float` rejects, and `-2147483648i` does not fit `i32`. Covered by
  `sarek/tests/e2e/test_int32_bitcasts.ml` over ±0, ±inf, denormals, max
  finite, a NaN payload and a signalling NaN: decode and encode (NaN rows
  checked for NaN-ness, since the host widens float32 to a double), a round
  trip inside one kernel and one through a float32 device buffer across two
  launches (bit-exact, signalling NaN included, on every GPU), a packed-record
  decode, and a `0x80000000l` literal operand. It passed 2026-10-10 on HIP, OpenCL,
  Vulkan and CUDA/PTX via ZLUDA (RX 7900 XTX and Ryzen 7950X iGPU), on Metal
  and OpenCL (Apple M4), and on Native and Interpreter; and by
  `sarek/tests/unit/test_int32_bitcasts_codegen.ml` (spellings, glslang/naga/
  clang validation with negative controls, refusals). Spec
  `specs/stdlib-bitcasts.md`.

### Changed

- The launch window (deferred launches, horizontal fusion) is on by default on
  CUDA (`CUDA/PTX`, `CUDA/C`), as it was on HIP, OpenCL and Vulkan (package H1).
  Horizontal fusion is bit-identical by construction. Evidence:
  `test_launch_window` executed on CUDA/PTX through ZLUDA on a gfx1100;
  CUDA/C uses the CUDA C generator HIP uses; neither ran on NVIDIA hardware.
  Metal stays opt-in until executed on Apple hardware: its fused source is
  checked as compiler output only. Metal has 31 buffer slots, so a group is not
  grown past 31 kernel argument slots (`Launch_window.arg_slot_limit`).
  `SAREK_LAUNCH_WINDOW=0` or `Execute.Window.set_backend_enabled` turns it off.
  `bench_launch_window` moved to `sarek/tests/e2e` and runs on every non-CPU
  device (`SPOC_BENCH_FRAMEWORK` narrows it; a backend outside the default set
  is opted in for the "on" runs). The `e2e-hip-bench` alias no longer includes
  it; `make bench-launch-window` runs it.

- Vulkan kernel launches are asynchronous. A launch appends its dispatch to the
  stream's open command buffer and returns; the buffer is submitted once it
  holds `SPOC_VULKAN_BATCH` dispatches (default 32), when another stream of the
  device launches, and at every synchronisation point. A stream has two command
  buffers, so one records while the other runs. Each dispatch has its own
  descriptor set (per-batch pools, so one kernel can appear several times in a
  batch) and a compute memory barrier before it. Every host access to device
  memory, `Stream.synchronize`, `Device.synchronize`, kernel-cache eviction
  and device destruction first complete the pending launches, and
  `Event.record` submits them. `Memory.free` (which GC finalisers call) never
  waits or blocks: the release is queued, and runs at once when the device is
  idle, otherwise at the next synchronisation point. Before, every launch recorded,
  submitted and waited for its own command buffer. A failed submission or wait
  now raises `kernel_launch_failed` naming the batch's kernels at the first
  wait, not at the launch. `SPOC_VULKAN_BATCH=1` restores submit-and-wait per
  launch. On an RX 7900 XTX (RADV, Mesa 26.1.4), `SPOC_VULKAN_BATCH=1` against
  the default: host cost of a launch 59 us against 6.6 us (3 arguments) and
  60 us against 8.2 us (10 arguments); a 24-layer decode-like chain 32 ms
  against 3.6 ms per token with the launch window off and 23 ms against 3.2 ms
  with it on (medians of 5 interleaved pairs, load average about 10,
  provisional). Results are bit-identical with the launch window on and off.
  Spec `specs/vulkan-async-launches.md`.

- The staged matrix rewrite (`Sarek_matrix_rewrite`) keeps B in shared memory
  transposed (`[n][k]`) and loads its fragments column-major, and each thread
  stages 8 consecutive k (reads, conversions, then contiguous stores). On
  gfx1100 a B fragment is two `ds_load_b128` instead of 16 two-byte loads and
  a staged group one `ds_store_b128`. Results are bit-identical; on an
  RX 7900 XTX the staged kernels are 1.3–3.5× faster (f32 as f16
  256×3072×3072: 659 → 269 µs; int32: 2664 → 766 µs; bf16: 1016 → 598 µs;
  median of 5 interleaved runs,
  `spoc-pipeline-logs/pkgF/F2-bench-matrix-rewrite-c42-summary.txt`) (C-42).

- The direct size class of the launch-time matrix rewrite
  (`Sarek_matrix_direct`) accepts column-major A, B and C: each fragment is
  loaded or stored in its operand's own layout. A float16 GEMM with B stored
  transposed now takes it when aligned: 256×1024×3072 from 401 to 71 µs,
  256×3072×3072 from 658 to 242 µs (RX 7900 XTX, median of 5 interleaved runs,
  `spoc-pipeline-logs/pkgF/F2-bench-matrix-rewrite-c41-summary.txt`). The
  CUDA/HIP fragment helpers take a 64-bit stride, so offset products no
  longer overflow `int` for strides above 2^31/15 elements.

- The CUDA/HIP generator drops `sarek_f32_barrier` at the f32 -> f16
  narrowings where it provably changes nothing (global review C-40,
  specs/f16-narrowing-barrier.md): the narrowed value is exact in f32 and its
  top operation cannot yield a zero of the wrong sign, or it is a sum or
  difference of two binary16 values. An exact product whose zero could be
  `-0` (e.g. a negative binary16 scale times an int8 quant) is emitted as
  `sarek_narrow_exact_product`, which keeps the fused instruction and restores
  the sign. Results stay bit-identical to the interpreter (16M-value sweeps on
  hip:0, `sarek-hip/test/test_hip_f16_narrowing_exact.ml`); Q8_0 with a
  binary16 scale, dequantised to LDS: 932 -> 182 us. Kernels that read their
  scales as f32 bits (Katra's) keep their barriers.

- A launch hashes its IR once instead of 4-5 times (R-41). The launch-path
  tables keyed by the IR's physical identity (capability verdicts and
  generated sources in `Execute`, the launch window's `ir_info`, the matrix
  plan) are typed slots of one record per IR (`sarek/execute/Ir_memo.ml`),
  still weak in the IR, and each domain keeps its last lookup, so
  consecutive launches of the same kernel hash nothing. `clear_launch_cache`
  and `Matrix_specialise.clear` now make entries stale by generation; a stale
  entry is replaced on the IR's next launch or dies with the IR.

- With logging off, a `Spoc_core.Log` call costs one integer compare (R-43):
  the level is tested first, against an atomic, and the enabled components
  are an atomic bit mask instead of a polymorphic `Hashtbl`.
  `Log.would_log level comp` lets a call site skip computing its arguments;
  the per-launch sites in `Transfer`, `Execute.get_device_buffer` and the
  OpenCL argument setters use it. `Log.log_level` (a `ref`) is gone: use
  `Log.set_level`.

- `Spoc_framework.Kernel_args` stores arguments in an array of options
  instead of a polymorphic `Hashtbl` (R-42): `set` and `validate_and_extract`
  no longer hash or compare polymorphically on every launch. Same API;
  indices below 0 or from 4096 up are kept in a sorted list, so a stray
  index never allocates storage of its size. One error text changed: two or
  more extra indices are now "unexpected indices", not "unexpected indexs".

- OpenCL CPU devices no longer share host memory implicitly. A vector whose
  host array happened to meet `CL_DEVICE_MEM_BASE_ADDR_ALIGN` used to get a
  `CL_MEM_USE_HOST_PTR` buffer, so whether a vector was copied depended on
  where `malloc` put it. Default buffers on every OpenCL device now have
  memory of their own; Native and Interpreter are unchanged (their device
  memory is the host array). `Framework_sig` `Memory.alloc_zero_copy` now
  means only "the host array is the device memory", and OpenCL returns
  `None`. Migration: code that relied on the implicit sharing (to avoid
  doubling memory for large arrays on the Intel CPU runtime, or that wrote the
  Bigarray directly and expected kernels to see it) creates the vector with
  `Host_ptr_buffer.of_bigarray` (or calls `Host_ptr_buffer.attach` before the
  first launch) from a 4096-aligned Bigarray, and calls
  `Host_ptr_buffer.publish_host_writes` after writing the Bigarray directly.
  Backend implementers add `alloc_host_ptr` (return
  `Error Host_ptr_unsupported` when there is none); hand-written
  `Memory.BUFFER` modules add `let host_link = Memory.Separate`. The test hook
  `Opencl_plugin_base.For_testing.zero_copy_on_any_device` is renamed
  `host_ptr_on_any_device`.

- Native kernels read float32 and int32 vector elements with an inline load
  and write them with `Vector.set`, without the `typed_value` round trip
  (`sarek/plugins/native/Native_vec_access.ml`). Each `v.(i)` used to go through
  the vector's `typed_value` interface and a polymorphic bigarray read, 9-10
  words allocated per access; on a multi-domain pool every minor collection
  stops all domains. Now 2 (float32) or 3 (int32) words, same bounds check,
  host sync and results. `benchmarks/bench_native_matvec.exe` (1024 x 3072
  Q8_0-style matvec, 8 selections, 28 domains on a 7950X): wave kernel 370 ->
  123 ms, barrier-free row kernel 167 -> 38 ms (medians, load ~38). OpenCL on
  the same CPU runs them in ~1.0 and ~0.6 ms. Test:
  `sarek/plugins/native/test/test_native_vec_access.ml`.


- Blocking driver calls release the OCaml runtime lock (R-24): device,
  stream and event syncs, synchronous copies and memsets, device frees, module
  loads and the hiprtc/NVRTC/shaderc/OpenCL compiles on HIP, CUDA, OpenCL and
  Vulkan (`vkWaitForFences`, `vkQueueWaitIdle`, `vkDeviceWaitIdle`). Before,
  a domain waiting in one of them stopped every other domain at its next minor
  collection for the whole wait. Measured on the RX 7900 XTX (5 interleaved
  runs, 200 ms `Device.synchronize`): minor collections completed by an
  allocating domain during the wait went from 0 / 2 / 1 (HIP / OpenCL / Vulkan
  medians) to 1082 / 1074 / 1094. None of these entry points takes an OCaml
  heap pointer. Memory reached through a raw address is no longer protected
  by the held lock: the caller of `Memory.host_ptr_to_device` /
  `device_to_host_ptr` (`Runtime.to_device_ptr` / `from_device_ptr`) must keep
  it allocated until the call returns (the pointer value itself is now kept
  reachable across the call). Metal followed in a later entry (its waits go
  through `objc_msgSend`). Not switched yet, though they can take milliseconds:
  `hipHostMalloc`/`hipHostFree`/`hipHostRegister`/`hipHostUnregister`,
  `hipStreamDestroy`, `hipEventDestroy`, `hipModuleUnload` and their CUDA
  counterparts (`cuMemAllocHost`, `cuMemFreeHost`, `cuMemHostRegister`,
  `cuMemHostUnregister`, `cuModuleUnload`), and OpenCL `clRelease*`. Test:
  `sarek/tests/e2e/test_blocking_ffi_releases_lock.ml`.
- CPU worker pools are retired when idle (R-66). Every live domain takes part
  in each minor collection, and the parallel interpreter and Native pools used
  to keep up to 95 idle domains (32-core host) for the life of the process.
  Pools are now created by their first launch only and retired, workers joined,
  once no launch has used them for `SAREK_CPU_POOL_IDLE_MS` (default 1000 ms);
  the next launch creates them again. Fission-queue dispatchers are retired the
  same way. `SAREK_CPU_POOL_MAX_DOMAINS` (or
  `Sarek_interp.Cpu_pool_lifecycle.set_max_domains`) caps each pool so a client
  can keep cores; `Cpu_pool_lifecycle.retire_idle_now` (also run `at_exit`)
  retires every idle pool at once. The interpreter and the Native barrier path
  now share one block pool (`Sarek_interp.Cpu_domain_pool`); each launch waits
  on its own task group, so concurrent launches no longer receive each other's
  errors (R-68, block pool). The barrier-free `run_parallel` path runs its
  thread ranges on that pool instead of spawning fresh domains per launch. The
  launching domain runs its own untaken tasks while it waits, so a launch from
  inside a block-pool kernel completes at any pool size. A
  pool whose creation hits OCaml's domain limit joins the workers it started
  before re-raising. Before, using every CPU pool on a 32-core host needed 129
  domains and failed with "failed to allocate domain"; now at most one pool of
  each kind (3 x 32), a dispatcher per fission queue and the reaper are live. Measured with
  `sarek/tests/unit/bench_cpu_pool_idle_gc.ml` (5 interleaved runs, load 7-23):
  minor-GC cost 1.5 s after one launch on each pool went from 715 us (530-810)
  to 258 us (248-313), against 283 us (279-289) in a process that never ran a
  kernel; live OS threads 194 -> 2. Results and the parallel decomposition are
  unchanged. Test: `sarek/tests/unit/test_cpu_pool_idle.ml`. Migration:
  `Sarek_ir_interp.DomainPool`, `get_pool`, `global_pool` and
  `resolve_domain_count` are gone; in `Sarek_cpu_runtime_pools`, `get_pool`,
  `global_pool` and `resolve_domain_count` are gone, `DomainPool.run n f`
  (run `f 0` .. `f (n - 1)`) replaces `submit`/`wait_all`, `with_fission_pool` / `with_parallel_pool` replace
  `get_fission_pool` / `get_parallel_pool`, and `LaunchQueue.retire_if_idle`
  replaces `shutdown`. The jsoo interpreter build runs parallel blocks inline.

- Fused kernels are named after a digest of their IR
  (`sarek_hfused_<md5>`, `sarek_efused_<md5>`) instead of a build counter,
  so a process that builds the same fusions in another order still hits the
  device compiler's on-disk cache (ROCm comgr on HIP). Two fused kernels
  whose IR differs get different names short of an MD5 collision; that is
  safe either way, because every compile cache keys on the source as well.

- The epilogue-fusion coverage check (`Launch_epilogue`) no longer
  enumerates every thread when store indices and guards are affine in the
  thread and work-group indices, also through division, modulo, masks and
  shifts by powers of two of values that are non-negative over the whole
  launch, guards not counted (a warp-per-row matrix-vector product, 4M
  threads, is proven; a guarded `(tid - 1) / 2` is left to enumeration):
  `Coverage_affine` proves coverage in a few microseconds (a 128x3584
  prefill pair: ~350 ms before, ~4 us after,
  `sarek/execute/test/bench_epilogue_coverage.ml`). This was the cold
  first-prompt cost seen with the launch window on: a cold Katra 0.8B prompt
  (242 tokens, RX 7900 XTX; measured with this change on top of 79f92f63)
  runs at 662 tok/s with the window on instead of 147, against 644 with
  epilogue fusion off and 706 with the window off; decode is unchanged
  (236 vs 232 tok/s). No pair of that prompt is enumerated any more.
  Enumeration, still used for other indices (loops, non-power-of-two
  division), uses bitsets instead of hash tables and is
  bounded to 2^20 threads for the pair (was 2^24 per launch); beyond it the
  pair stays unfused, and an enumerated pair still costs milliseconds (6 ms
  for 2 x 131072 threads).

- `Spoc_core.Memory.bigarray_void_ptr` is replaced by
  `Memory.borrow_bigarray_ptr`, whose pointer need not root the bigarray: the
  caller keeps it reachable, on transfer paths with a trailing
  `ignore (Sys.opaque_identity ba)` after the FFI call (Memory, Transfer,
  CUDA, HIP, Metal, OpenCL, Vulkan, Native), and elsewhere through the owning
  vector (raw addresses, `Vector.to_ctypes_ptr`, OpenCL
  `CL_MEM_USE_HOST_PTR`). `Memory.with_bigarray_ptr ba f` does the same around
  a callback and is used on the cold OpenCL zero-copy allocation only, since
  a capturing callback allocates a closure. The f16 host pointer no longer
  rebuilds a managed ctypes pointer with `Obj.repr`, and no `Obj` remains in
  the tree. The `gc_roots` test in `test_float16.ml` covers
  `with_bigarray_ptr` and the borrow-then-`opaque_identity` pattern, not each
  backend call site. Out-of-tree callers of `bigarray_void_ptr` must switch to
  `borrow_bigarray_ptr` and keep the bigarray reachable across their call.

- Intrinsics that no backend lowers soundly are refused at PPX time, with a
  located error. `warp_shuffle`, `warp_shuffle_down`, `warp_shuffle_up`,
  `warp_shuffle_xor`, `warp_vote_all`, `warp_vote_any`, `warp_ballot`,
  `warp_active_mask` and `warp_size` type-checked but no backend lowers them.
  Before this change a PPX kernel using `warp_shuffle` (or another function
  form) failed only when OCaml compiled the generated native code (`Unbound
  value Gpu.warp_shuffle`), and one using `warp_size` failed inside the PPX
  with an internal `Failure("Unknown intrinsic constant: warp_size")`
  (`warp_size` was typed `int32`, with no lowering and no single value: the
  warp width varies by vendor and device). They are now refused as "not
  lowered by any backend yet" until warp-scope supersteps (roadmap S1) lower
  them. `return_unit` compiled silently before; it is now refused with advice
  to restructure with if/else: CUDA emitted `return((void)0);`, which skips
  every later barrier of that thread, and the native backend ignored it. The
  refusal keys on the resolved intrinsic (`Sarek_unlowered.table`, one table),
  so `Gpu.warp_ballot` and `let open Std in warp_shuffle ...` are refused too,
  while a variable or kernel-local function that happens to be named
  `warp_size` is not (a local function now shadows a refused intrinsic of the
  same name; other intrinsic names keep their existing resolution order). The
  table entries and their variance/convergence classes in
  `Sarek_core_primitives` are unchanged and the convergence checker is still
  unit-tested on them.
  `test_warp_diverged` now gets the refusal rather than the convergence error.
- The automatic float matrix rewrite inside `Execute.run_vectors` now has its
  own explicit opt-in, `Sarek_coopmat.accept_automatic_float_rewrite ~reason`
  (+ `automatic_float_rewrite_reason`; no env var, one-way, first reason kept).
  `accept_relaxed_accumulation` / `accept_operand_rounding` keep the meaning
  they had before the automatic rewrite: they admit matrix-unit configurations
  for code that asks for them (`Sarek_gemm.Coopmat`, `Sarek_matrix.run`), and
  no longer enable the automatic rewrite of a program's own float kernels.
  Without the new opt-in such a launch runs the original kernel and the
  `Original` reason (and `SAREK_MATRIX_REPORT=1`) names the missing call.
  int32 is unchanged (rewritten by default); `Sarek_matrix.run` does not need
  the new opt-in. Migration: programs that called `accept_relaxed_accumulation`
  to get the automatic float rewrite must also call
  `accept_automatic_float_rewrite ~reason` (this changes which kernel runs, not
  an API signature). `SAREK_MATRIX_REWRITE` is parsed strictly: `off`/`0`/
  `false`/`no` (any case) switch the rewrite off, which silently meant `auto`
  before; an unrecognised value warns on stderr and gives `auto`.

- The stream keep-alive and kernel-argument retention no longer use `Obj`.
  `Transfer.retain_on_stream` takes `~keep:(Transfer.Keep v)` (the new
  existential `Transfer.keep`) instead of an `Obj.t`; the stream ledger,
  the ranged copies and registered host buffers root their operands through
  typed values; the HIP and CUDA backends retain a launch's argument blocks in
  a record of their ctypes types; and `Execute`'s source memo keys the backend
  by its `generate_source` closure at its own type, still compared physically.
  Behaviour is unchanged; a new device-buffer test checks that a stream copy's
  host buffer (either direction) and a retained value survive `Gc.full_major`
  until the stream is synchronised.
- `create_array` is checked at PPX time on every backend, Native included. It
  used to be accepted everywhere and mean different things: `create_array n
  Global` became a per-thread private array on CUDA/OpenCL/Metal/GLSL (and was
  refused by PTX/WGSL), and `create_array n Local` with a runtime `n` became a
  variable-length array that OpenCL C and MSL reject. Now `Global` is a located
  error (take a kernel vector parameter instead) and the size of a `Local` or
  `Shared` array must be a positive compile-time constant (an int32 literal, or
  literals combined with `+ - *`, folded to a literal before any backend sees
  it; every intermediate result must fit in int32). The same rule applies to an
  explicit `let%shared` size (`let%shared (t : int32) = n in` with a runtime
  `n` is now refused); the size-less `let%shared (t : int32) = ()` form is
  unchanged. The CUDA, OpenCL, Metal and GLSL emitters also refuse a hand-built
  `EArrayCreate(_, _, Global)` instead of declaring a private array, and
  `EArrayLen` on a created array is emitted as its literal extent (it named a
  length symbol only vector parameters have).
  **Migration**: replace a runtime size by a constant upper bound, e.g.
  `create_array n Local` becomes `create_array 16l Local`, and index only the
  first `n` slots. This requires `n <= 16`: guard it, or choose a bound that
  covers the largest `n` the kernel can see. Replace `create_array n Global` by
  a vector parameter. The `Pexp_array` hint no longer recommends `Global`.
- The `CM_decl` IR docstring now states that a declared cooperative-matrix
  fragment is zero-initialised on every backend (as `specs/coopmat-hip-wmma.md`
  already said); OpenCL, Metal, WGSL and PTX refuse cooperative matrices rather
  than declare one uninitialised.

- `lsr` compiles to one unsigned shift on every backend. It is now its own IR
  operator (`Lshr32`/`Lshr64`) instead of a 4–5 operation expansion over the
  arithmetic shift: CUDA/HIP emits `((int)((unsigned int)(a) >> (b)))`,
  OpenCL and Metal `((int)((uint)(a) >> (b)))` (64-bit: `unsigned long long`
  / `ulong`), GLSL `int(uint(a) >> uint(b))`, WGSL
  `bitcast<i32>(bitcast<u32>(a) >> u32(b))`, PTX `shr.b32`/`shr.b64`. Results
  are unchanged for shift amounts 0..width-1 (others stay unspecified, as in
  OCaml), and an `lsr` operand no longer has to be a variable or a constant.
  spoc_llm's q5_K matvec written with `lsr`, on an RX 7900 XTX: 31.6 -> 23.9 us
  (hot, median of 3) and 36.3 -> 31.1 us (cold). On gfx1100 the old expansion
  kept clang from unrolling the inner loop (193 VALU, 2 `global_load_b128`);
  with the native shift clang unrolls it (274 VALU, 12 `b128`), the same code
  as the hand-written `asr`-and-mask form. Spec `specs/codegen-lsr-unroll.md`.

- `for` loops with constant bounds and a trip count of at most 16 get an
  unroll hint by default (`#pragma unroll` in CUDA/HIP, OpenCL and Metal,
  `[[unroll]]` with `GL_EXT_control_flow_attributes` in GLSL, nothing in WGSL
  and PTX), as long as the nest they head expands to at most 64 copies. Results
  are unchanged. An explicit `pragma` on the loop takes precedence
  (`pragma ["unroll 1"]` opts out) and `SAREK_AUTO_UNROLL=0`, set before a
  kernel's first launch, disables the hint.
  On spoc_llm's current matvec kernels the hint changes nothing on gfx1100:
  clang -O3 already unrolls those loops once `lsr` is a native shift.

- Metal kernel launches are asynchronous. Dispatches are appended to one open
  command buffer per queue. At each launch it is committed if the queue's
  previous command buffer has completed or once it holds `SPOC_METAL_BATCH`
  dispatches (default 32); otherwise the next launch, wait, event record or
  synchronisation commits it. Every
  host access to a buffer waits for the pending command buffers that bind it,
  and a launch waits for pending work on other queues that binds its buffers.
  Before, every launch committed its own command buffer and waited for it. A
  failed command buffer now raises `kernel_launch_failed` (it was ignored), at
  the first wait that observes it, naming every kernel the command buffer ran.
  `Event.elapsed` measures GPU end times (it subtracted host record times, and
  a fresh start event gave about the epoch).
  Measured on an Apple M4: 195 us per launch before, 4.3 us (3 arguments) /
  8.0 us (10 arguments) after (`bench_metal_launch_overhead.exe`, 10000
  launches); spoc_llm Qwen3.5-0.8B Q8_0 greedy decode 8.6 tok/s before, 67–70
  tok/s after, same tokens.

- Intrinsic libraries outside sarek are a supported extension point
  (`sarek/ppx_intrinsic/README.md`, "Using intrinsics from another library"):
  declare the library `(kind ppx_rewriter)` and list it in
  `(pps sarek.ppx <lib>)`. Covered end to end by `sarek/tests/downstream/`,
  which runs on Native and Interpreter, and on HIP, CUDA C, OpenCL and Metal
  when those devices are present. The PTX, GLSL and WGSL backends do not
  consult the intrinsic registry. They refuse such kernels with an error that
  names the intrinsic.
- `let%sarek_intrinsic` also registers a host implementation
  (`Sarek_registry.register_host_impl`) for every all-scalar signature. The
  Interpreter falls back to it, so it now runs intrinsics declared outside its
  built-in tables. Before, it raised `Unknown intrinsic`.
- Device-only vectors, ranged and stream-ordered copies, pinned host memory,
  kernel launches on a stream, and events (`Spoc_core.Transfer`,
  `Execute.run`/`run_vectors ?stream`; spec
  `specs/device-buffers-async-copy.md`). `create_device_only` allocates a
  vector with no host mirror; every host access to it refuses instead of
  allocating. `copy_to_device`, `copy_bytes_to_device`, `copy_ptr_to_device`
  and `copy_to_host` copy a range at an offset, synchronously or on a
  stream, and SPOC keeps the operands alive until the stream or a later
  event completes. `alloc_pinned`/`free_pinned` and
  `register_host`/`unregister_host` give page-locked Bigarrays. Implemented
  on HIP and CUDA; synchronous memmove semantics on Native and Interpreter;
  refused with `Feature_not_supported` on OpenCL, Vulkan, Metal and WebGPU.
  Measured on an RX 7900 XTX (ROCm 7.2.4), 256 MiB host->device into a
  device-only vector, best of 5: pageable 28.1 GB/s, pinned 28.6 GB/s
  synchronous; 27.5 / 28.3 GB/s on a stream
  (`sarek-hip/test/bench_hip_pinned_copy.exe`, information only).

### Changed

- Repeated launches of the same kernel are cheaper (spec
  `specs/cheap-repeated-launches.md`). `Execute.run`/`run_vectors` memoise,
  per physical kernel IR, the device capability check and the generated
  source (keyed by backend, block and SoA parameters); failures are never
  memoised and the argument checks still run on every launch.
  `Compile_cache.make_key` memoises keys per physical source string. The HIP
  launch marshals all arguments into one buffer instead of one allocation
  per argument. New `Execute.clear_launch_cache` drops the memo; backend
  cache clears and device teardown call it. Host cost per `run_vectors`
  launch on an RX 7900 XTX (`sarek-hip/test/bench_launch_overhead.exe`,
  10 000 launches, no sync, tartarus, 2026-10-10, base beb81321 vs this
  change): 3 arguments 11.4 -> 4.4 us, 10 arguments 17.4 -> 6.5 us. Native
  is unchanged (its launch cost is execution).

- `let%sarek_intrinsic` registers an intrinsic under its library's module path,
  using the `library-name` cookie dune passes to the PPX, instead of guessing
  from the source directory name. sarek's own libraries keep their paths. A
  downstream library in a directory not named after it (for example
  `src/intrinsics/` holding `foo_intrinsics`) used to register as `["Bits"]`
  rather than `["Foo_intrinsics"; "Bits"]`.
- **Breaking:** a qualified name in a kernel (`M.f`) that the kernel PPX
  cannot resolve is now a compile error: "Unknown kernel function or
  constant M.f ...". The error names the fix and lists the names the PPX knows
  under `M`. Before, the PPX accepted it untyped. Native ran it as an OCaml
  call, but every source backend printed it verbatim into device code, so it
  failed only at JIT time. Three in-tree sites depended on this:
  `test_visibility_private` now links `sarek.visibility` into its PPX, and
  `test_debug_native` and the `reduction_max` benchmark kernel now spell their
  values in Sarek.
- **Breaking:** `Framework_sig.BACKEND` gains `Async` (an `ASYNC` module), `wrap_stream`,
  `unwrap_stream`, `wrap_event`, `unwrap_event` and `run_source_on_stream`;
  `PLUGIN_BASE` gains `Async`. An out-of-tree backend that does not provide
  them adds `module Async = Framework_sig.Async_refused (...)`,
  `include Framework_sig.Make_refs (...)` and a `run_source_on_stream` that
  calls `Framework_sig.refuse_stream_launch` (about 20 lines; see
  `sarek-opencl/Opencl_plugin.ml`).
- **Breaking:** `Vector.host_storage` has a third constructor, `Device_only_storage`. Code
  that matches on `vec.host` must handle it (every in-tree site refuses).
- **Breaking:** `Transfer.STREAM` gains `stream_ref` and `state`; `destroy_stream` now
  synchronises the stream before destroying it.
- **Breaking:** `Execute` refuses a vector longer than `Int32.max_int` elements at launch;
  the injected length used to wrap silently.
- **Breaking:** `Sarek_tuple_vec.descriptor_by_name` is removed. It returned a
  tuple `custom_type` at whatever type the call site asked for, so a shape that
  did not match the element type type-checked and read the vector bytes at the
  wrong layout (a `(float32 * int32)` descriptor taken at `(int64 * float)`
  wrote `(42L, 3.5)` and read back `(4343, 4.66e-310)`). Generated Native code
  now calls the typed, memoised `Sarek_tuple_vec.pair` / `triple`, which
  returns the descriptor the host built (same `Type_id`) and builds it if the
  host has not; a tuple-shape record name with an unknown component stops the
  ppx expansion. The registry is typed (a GADT shape key per slot, no `Obj`)
  and its reads are lock-free. That per-element call costs 11 ns against
  32 ns for the old `descriptor_by_name` on one domain, and 14-19 ns against
  934 ns per call per domain with 8 domains calling at once (tartarus,
  `taskset -c 2-9`, best of 5 x 2 000 000 calls, 2026-10-11). Tests:
  `test_tuple_native_descriptor`, `test_tuple_vec_registry`.

### Fixed

- Vector ids are allocated with `Atomic.fetch_and_add`. The plain `ref` +
  `incr` handed out duplicate ids when vectors were created from several
  domains (4 domains x 200 000 creations: 450 907 duplicates of 800 000), and
  the id is buffer identity for launch-window conflicts, epilogue fusion and
  the transfer keep-alive ledger. Test: `test_vector_ids`.
- `Memory.host_to_device` / `device_to_host` size the copy from the bigarray's
  own bytes and raise `Invalid_argument` when its element size differs from the
  buffer's or it holds more bytes than the buffer (a float32 bigarray used to
  type-check against a float64 buffer and over-read / overflow the host
  array). `Memory.device_to_device` refuses a source larger than the
  destination. D2D still bounces through host memory: the `BUFFER` signature
  has no backend device-to-device copy. Test: `test_memory_sizing`.
- Interpreter (the oracle), three corrections:
  - Cooperative-matrix operations execute once per 32-lane subgroup, on a
    fragment owned by the subgroup, instead of once per invocation. In-place
    `D = A*B + D` (C loaded from the array D is stored to) gave `C + blockDim *
    A*B` (512 at block 32, 1024 at block 64 for a 16x16x16 u8 tile; 16 is
    right); `Sarek_gemm.Coopmat` with `c` and `d` the same array was wrong the
    same way. The interpreter assumes what the cooperative-matrix specification
    requires: every lane of the subgroup reaches the operation and its buffer,
    index and stride are uniform across it.
  - Two kernel arguments bound to the same vector share one value array
    (`Execute.run_interpreter_vectors`, `Sarek_ir_interp.run_kernel_with_args`).
    `x.(i) <- y.(i) + 7` with `x` and `y` the same vector left `x` unchanged
    (the two private copies were written back in an order that depended on the
    entry point). This covers `Execute.run_vectors` on an Interpreter device
    too (`run_kernel_with_exec_args`, matched by the vector wrapper's stable
    `underlying_type_id` and physical `underlying`). Not covered: the
    interpreter plugin's own device-buffer wrapper, which mints a fresh type id
    per instance and is not reachable from launches built from host vectors.
  - The BSP deadlock detector could never fire (a resumed thread was counted as
    completed). It now raises `BSP_deadlock` when threads wait at a block barrier that
    finished threads never reach, and does not fire on successive barriers or on
    threads parked at a warp barrier (a new `Warp_barrier` effect).
- The matrix rewrite no longer narrows int64 indices to int32 (compiler review
  C-11). A GEMM whose index, bound or guard (the guards around the product
  and its store) involves an `int64` constant, variable or cast used to be recognised with the casts dropped (`Int64.to_int`)
  and rewritten with int32 addresses, which overflow past 2^31 elements: out of
  bounds or wrong results, by default on HIP gfx11. `Sarek_ir_affine.of_expr`
  now returns the new `Wide_index` error, the pattern reports the new reason
  `Int64_arithmetic` ("uses int64 arithmetic"), and `Sarek_matrix.run` returns
  `Original` with that reason, so the advice never claims matrix units for such
  a kernel. `Sarek_ir_eval_int.eval` returns `None` for an int64 constant,
  variable or cast instead of truncating it. int32-indexed products are
  unchanged. Migration: code that matches exhaustively on
  `Sarek_ir_affine.error` or `Sarek_matrix_pattern.reason` gains one case each;
  a kernel that wants the rewrite must index with int32.
- OpenCL zero-copy buffers (`CL_MEM_USE_HOST_PTR`, chosen for CPU devices) had
  no-op host transfers, so a host read raced the asynchronous kernel and a host
  write raced kernels still reading; on an implementation that keeps a private
  copy the result never reached the host at all. Host access now goes through a
  blocking `clEnqueueMapBuffer` / `clEnqueueUnmapMemObject` (and a `clFinish`),
  and `USE_HOST_PTR` is used only for a pointer aligned to
  `CL_DEVICE_MEM_BASE_ADDR_ALIGN`; other arrays take an ordinary buffer.
  OpenCL's `is_zero_copy` is now `false` (a USE_HOST_PTR buffer is not "no
  transfers needed"), so `Transfer` forwards host copies to the backend.
  Regression tests `sarek-opencl/test/test_opencl_zero_copy_sync.ml` (backend)
  and `sarek/tests/e2e/test_opencl_zero_copy_vector.ml` (Vector path), forced
  onto GPU devices by the test-only `Opencl_plugin_base.For_testing` hook.
- `test_launch_window`: on every device three plain runs must agree bit for
  bit, and a disagreement fails. The window-on comparison is made only on
  devices that defer launches; on the others (every CPU device) it is skipped,
  with the skip counted in the summary line, because window on and off take the
  same path there. No barrier race was found in the generated OpenCL for the
  `mv_wave` / `mv_row` kernels; the plain-path nondeterminism seen on HIP on
  the 7950X iGPU remains open (possibly #427).

- Device selection on three runtime paths. `Spoc_core.Kernel.launch` (so
  `Runtime.run` and `Runtime.run_kernel`, including compile-cache hits) now
  selects the kernel's device before launching: with two HIP devices a launch of
  device A's module with device B current failed with `hipErrorInvalidDevice`.
  `Profiling` events and timers select their device before creating, recording,
  synchronising and destroying (HIP and CUDA events and the default stream belong
  to the current device). `Cuda_api.Memory.free`, which runs from GC finalisers,
  restores the previously selected context, as `Hip_api.Memory.free` already did.
  Regression tests: `test_hip_launch_selects_device`,
  `test_hip_profiling_selects_device` (`e2e-hip`), `test_cuda_free_keeps_current`;
  each needs two devices of one backend and self-skips otherwise.
- Vulkan events and fences no longer hang the host. `Event.record` now
  submits the event's fence on the device's compute queue (it did nothing, so
  `Event.synchronize`, `Transfer.synchronize_event` and
  `Profiling.timer_elapsed_ms` waited forever); an event that was never
  recorded synchronizes at once, and `Event.destroy` waits for a pending
  record before destroying the fence. `Event.elapsed` raises
  `Feature_not_supported` instead of answering 0.0, since the backend issues
  no timestamp queries. A launch refused on the host (for example a
  push-constant block over 128 bytes) no longer resets the stream fence first,
  which made the next launch on that stream hang: validation runs before the
  fence is touched. Each fence is waited on only after a successful
  submission, so a failed `vkQueueSubmit` (launch or event record) leaves
  nothing to wait for; `test_vulkan_fence_sync` covers this by fault
  injection. Fence waits on submitted work stay unbounded (stopping early
  would let a caller free memory the GPU still uses) and log a warning every
  60 s; `SPOC_VULKAN_FENCE_TIMEOUT_S` bounds them, raising
  `Synchronization_failed`, for tests that detect hangs. Submissions and
  waits for idle on a device's queue are serialized by a per-device lock; this
  does not make the backend multi-domain safe. Several domains may launch
  only on distinct streams created with `Stream.create`, after a first use
  from one domain; the default stream is single-domain, and a kernel object
  must not be launched from two domains at once.
- GLSL `CM_decl` emitted a bare `coopmat<..> x;` with undefined contents,
  while the Interpreter, HIP and CUDA zero the fragment. It now emits
  `coopmat<..> x = coopmat<..>(0);` (glslangValidator, Vulkan 1.3: exit 0).
- WGSL lowers a constant-size `create_array n Local` to a function-scope
  `var a : array<T, N>;` instead of failing with an internal "local dynamic
  arrays unsupported" error (naga: validates).

- hiprtc and nvrtc option arrays: `CArray.of_list string` let the C copies of
  the option strings be freed before the compile call. Under a different
  allocation pattern hiprtc read a garbage option ("error reading '8{...'"),
  and `test_hip_f16` failed deterministically with the launch window on. The
  strings are now explicit char arrays, kept alive until the call returns.

- Ranged-copy range checks no longer overflow: `copy_to_device`,
  `copy_bytes_to_device`, `copy_ptr_to_device`, `copy_to_host`,
  `copy_bytes_to_host` and `stage_to_device` compared `offset + size` with the
  total, which wraps for an offset near `max_int` and let the copy through
  (Native segfaulted reading `max_int - 2` bytes past the buffer). They now
  compare `offset > total - size`. The synchronous paths also keep the vector
  alive across the backend call, not only the Bigarray, so its buffer cannot
  be finalised mid-copy.
- CUDA: NVRTC is found on a CUDA 13 install that has only the soname
  `libnvrtc.so.13` (no development symlink). The loader tries
  `Cuda_nvrtc.nvrtc_candidates` in order: `libnvrtc.so`, then `.so.13`, `.so.12`,
  `.so.11`, then the macOS and Windows names (`nvrtc64_130_0.dll` added). Test
  `sarek-cuda/test/test_cuda_nvrtc_candidates.ml`. Trade-off: a host with a
  CUDA 12 driver, both `.so.13` and `.so.12`, and no `libnvrtc.so` symlink now
  gets NVRTC 13, whose PTX the 12.x driver rejects at module load and which has
  no compute_50..72 targets; install the development symlink pointing at the
  toolkit that matches the driver.
- `docs/optimization/l15b-mma-probe.md` no longer says ZLUDA does not execute
  `mma.sync`: on ZLUDA v7-preview.3 (RX 7900 XTX) m16n8k16 f32.f16 runs as a
  dot-product emulation (inexact, within the `γ_K` bound) and m16n8k32 s8 runs
  bit-exact; u8, satfinite, m16n8k8, f16 accumulate and every `wmma` form are not
  translated. `test_ptx_mma_probe` now checks each form separately against a CPU
  reference, with a control kernel, and declares the PTX ISA version the
  device's SM needs. Its statuses are ZLUDA's, not NVIDIA's.
- HIP: freeing a device buffer no longer leaves its device selected. The free
  runs from GC finalisers, so with two HIP devices a finaliser could fire
  inside a launch on the other device and make it fail with
  `hipErrorInvalidDevice`. Regression test
  `sarek-hip/test/test_hip_free_keeps_current.ml`.

- HIP `Stream.synchronize` now selects the stream's device first, as the CUDA
  backend has since backlog-225: synchronising `default_stream d2` while d1
  was current drained d1 and released d2's kernel-argument keep-alives.

- The Native and Interpreter CPU backends are now installed, as
  `sarek.native` and `sarek.interpreter`. They had no `public_name`, so code
  outside this repository could not link them, although the `sarek` package
  description and both plugin READMEs already advertised them. A cram test
  (`sarek/plugins/test/install_consumer.t`) builds an out-of-tree project
  whose OCAMLPATH is the install tree and links both.
- `benchmarks/descriptions/generated/reduction_max_generated.md`: all four
  device sources contained `Float.neg_infinity`, which is not valid device
  code. They now contain `-FLT_MAX`.

## 2026-07

### Added

- Direct PTX emitter completion: records/variants with aligned C-ABI layout,
  match, helper-function inlining, static + dynamic `.shared`, `.local`
  arrays, full atomic set (CAS, wrapping inc/dec, 64-bit add/exch), f64
  softmath (trig, exp/log family, hypot, …), scalar cast matrix
- `Sarek_worklist` — portable dynamic-parallelism (work-queue) library
- `Sarek_gemm` — portable tiled SGEMM library for shared-memory backends
- `Sarek_df64` — float-float extended-precision arithmetic in pure Sarek
- Portable float64: `G`-suffix float64 literals, `sarek.real64` host
  plumbing, complete interpreter float64 math oracle
- Tier-0 defunctionalization pass (first-class kernel-local functions)
- Tuple-typed vectors on device and Native paths (shared shape registry)
- Static tag erasure for kernel-local variant slots
- Aligned C-ABI aggregate host layout (L8), with the `PtxLayout.v` formal
  model restated for the aligned ABI (0 admits)
- float32 `fma` intrinsic on all backends; GLSL `precise` qualifier on
  float locals
- CI installs `cuda-nvdisasm-12-6` and `ci/assert-toolchain.sh` asserts
  `nvdisasm` (version + a ptxas→nvdisasm probe). It was absent from the built
  image, so `test_cuda_f16_sass` — the gate the NVIDIA f16 guarantee rests on,
  and the only one that could surface a 12.6-vs-13.3 ptxas divergence —
  self-skipped in CI while reporting green.
- `docs/fp-contraction-policy.md` — cross-backend floating-point contraction
  policy: what each backend may contract, what actually prevents it, and
  whether that mechanism is verified or merely believed. The interpreter is
  named as the cross-backend oracle. Linked from every site that previously
  carried an ad-hoc contraction comment. §7 ("what cannot be verified without
  NVIDIA hardware") is now partly closed: an f16 kernel HAS been executed on
  an NVIDIA GPU — GTX 1070 Max-Q, sm_61, CUDA 12.9, driver 580.119.02 —
  agreeing with the interpreter on all 63488 finite binary16 inputs, with a
  liveness control (63085 mismatches on deliberately mismatched kernels)
  proving the sweep can go red. That also settles the tie-rounding question
  the section named, exercises the *driver's* ptxas rather than the offline
  one, adds a second toolkit version (12.9, between CI's 12.6 and the
  document's 13.3), and independently confirms on hardware that the CUDA
  barrier removed in #110 was inert. sm_61 is below the sm_75…sm_121 range
  the host-side sweeps cover, so it is a new sample, not a repeat.
- FP-conformance guard on the CUDA/nvrtc path: `-use_fast_math`, `-ftz=true`,
  `--prec-div=false` and `--prec-sqrt=false` are now REJECTED at the point an
  option array reaches `nvrtcCompileProgram` (`--fmad=true` warns). The guard
  screens the whole ARRAY, because nvrtc accepts an option and its value as two
  elements: `["--ftz"; "true"]` compiled a subnormal-flushing kernel past a
  per-element check (confirmed against libnvrtc 13.3, `.ftz` in the emitted
  PTX). A value-taking name consumes the next element and is fail-closed. They flush
  binary32 subnormals or downgrade div/sqrt, and no later flag undoes that.
  `sarek-cuda/test/test_cuda_fp_conformance.ml` reproduces the hazard
  host-side (`-ftz=true` turns `FMUL`/`FADD` into `FMUL.FTZ`/`FADD.FTZ` at
  sm_90, CUDA 13.3) and each of its four cases was proved red by mutation.
- `sarek/tests/e2e/test_vulkan_no_contraction.ml` (on the `e2e-gpu` alias, so
  it executes rather than merely building) — settles whether a Vulkan driver
  honours SPIR-V `NoContraction` (#126). Same shader compiled twice,
  differing only by `precise`, run on the same device/driver/process; the
  contracted target is taken from the device's own `fma()` rather than an IEEE
  model, because RADV's `fma` is not correctly rounded and a modelled target
  can make a genuinely contracted result read as clean. **Measured on RX 7900
  XTX (RADV NAVI31) and the Raphael iGPU, Mesa 26.1.4-arch3.1: 0 of 7
  contraction shapes contracted with or without `precise`, and the emitted RDNA
  ISA is opcode-identical between the two builds.** Both in-tree claims are
  refuted: RADV neither ignores the decoration nor needs it. Mesa ANV remains
  unmeasured — no Intel GPU on this machine.
- `sarek-hip/test/test_hip_f16_shapes.ml` + `scripts/f16_shape_isa_audit.sh` —
  exhaustive f16 expression-shape audit for AMDGPU fusion demotion (#106). All
  20 shapes the DSL can emit, each swept over all 63488 finite binary16 inputs,
  on gfx1100 and gfx1036: **0 disagreements as shipped**. Removing the opacity
  barrier breaks 9 of 20 and reproduces the original 620 exactly — the harness
  was calibrated against that known positive before its nulls were trusted, and
  it fails closed if the barrier-removed control stops going red. Disassembly
  additionally shows *four* demotion opcodes, not two (`v_mul_f16` and
  `v_sub_f16` are new), and three shapes that are demoted in machine code yet
  numerically clean — which a numeric-only audit would have mis-reported as
  unaffected. See `docs/optimization/amdgpu-f16-fusion-shape-audit.md`.
- ZLUDA/AMD support for the CUDA/PTX backend (PTX launch ABI fix)
- `SAREK_REQUIRE_PTX` for `test_soa_emitter_equiv`: set it to exactly `1` and
  the test fails when no enumerated device reports the `CUDA/PTX` framework,
  instead of exiting 0 with the SoA legs either skipped or reported as PASSED
  with a `(SoA skipped: non-PTX)` note. Nothing in the repository's dune
  rules, Makefile targets or CI jobs sets it, and unset (or any value other
  than `1`) nothing changes, so an absent device still reads as skipped
  rather than as a pass or a failure. It asserts enumeration only — whether
  the SoA cases pass on that device is what the per-case rows say.
  Opt-in-by-hand; see `gh-pages/docs/device_selection.md`.
- **KNOWN RESIDUAL (backlog-225, not fixed).** `test_soa_emitter_equiv`'s
  `check_roundtrip` and the `Vector.fill` row of `check_relaunch` have each
  been seen, once, to return `y` equal to the un-doubled host value on
  ZLUDA's CUDA/PTX view of this host's devices — a leaf write that appears
  not to have landed before read-back. 2 occurrences noticed over "some
  2,500" executions of the test binary; a separate non-reproduction campaign
  afterward ran 2,443 further executions (2,000 + 300 + 40 concurrent + 40 on
  a cold ZLUDA cache + 3 under `dune runtest --force` + 60 isolated), all
  zero, which does not reconcile arithmetically against the ~2,500 figure —
  whether the 2 occurrences fall inside or outside that 2,443 cannot be
  determined from the record. No root cause established; nothing here
  claims a fix. `Transfer.flush` was added on the `check_roundtrip` leg as a
  defensive measure, not offered as the fix — the `check_relaunch` occurrence
  was on a leg that already synchronizes. Also found and filed but NOT
  implicated as the cause (both failing legs already call
  `Device.set_current` correctly via `Device.synchronize`): `Cuda_api.ml`'s
  `Stream.synchronize` was the only function in its module omitting
  `Device.set_current` before touching a stream/device (fixed here), and
  `Cuda_plugin_base.ml`'s `current_device` is a plain global ref while
  `Cuda_api.ml` underneath it deliberately uses a `Domain.DLS` key for the
  same state (not fixed; filed). See the full comment at
  `sarek/tests/e2e/test_soa_emitter_equiv.ml:703` and
  `docs/optimization/tier1b-emitter-soa-handoff.md`'s backlog-225 status
  block.
- T3-SEMANTIC milestone lock for both formal projects; conformance +
  mutation tests wired into `dune runtest`

### Changed

- **BREAKING (`Sarek_ir_codegen`, shipped in `spoc.opam`'s `spoc.ir` library).**
  The record-declaration ordering surface added for backlog-203 is replaced by
  one that also orders variants (see the backlog-211 entry under Fixed).
  Removed: `sort_record_types_by_dependency`, `referenced_record_names`,
  `gen_record_typedefs`. Renamed: `Record_type_cycle` → `Type_decl_cycle`, which
  now reports a cycle with either kind on it. Added: `type_decl`, `tie_break`,
  `sort_type_decls_by_dependency`, `referenced_type_names`, `gen_type_decls`,
  `gen_c_type_decls`. No compatibility aliases, deliberately: an alias for
  `sort_record_types_by_dependency` would still compile inside a per-kind
  emission loop and silently reintroduce the cross-kind bug. The three C-family
  backends also lose their local `gen_variant_def` wrappers —
  `gen_c_type_decls` takes `~constructor_prefix` directly. Nothing outside
  `spoc/ir` and its tests referenced any of the removed names.
- Breaking: `Sarek_type_helpers.HELPERS` gained a required `val field_names :
  string list` — the immediate field names in `to_values` order. An in-place
  record-field store needs a field's POSITION in the `VRecord`'s value array,
  and nothing in the previous signature could supply one: `get_field` only
  reads, and `to_values`/`from_value` round-trip through the OCaml record and
  therefore copy. For a PPX-generated RECORD helper the correspondence holds by
  construction: `field_names` comes from the same label list that generates
  `to_values` and `get_field`, so the three cannot drift. Two carve-outs, because
  the justification for a breaking change should not be stated more widely than
  it holds: the PPX emits `field_names = []` for a VARIANT (no named fields, so
  every lookup returns `None` and a field store on one is refused by name rather
  than writing an arbitrary slot), and one in-tree implementor is *not* PPX-
  generated — the hand-written test mock in
  `sarek/sarek/test/test_sarek_type_helpers.ml`, which maintains the order
  manually and whose own test checks `field_index` against `to_values`' output
  rather than against hardcoded slots so that a drift there fails. A hand-written
  implementation outside this repository has to add the list, and carries the
  ordering obligation itself. There is no non-breaking spelling — a module type
  cannot carry a default, and the index is not derivable from the rest of the
  signature.

- **BREAKING (backlog-185/200): all five source-emitting backends now REFUSE a
  `[%native]` block instead of emitting for it.** Before this change CUDA,
  OpenCL and Metal raised on `SNative` unless a prior transpile had left a tag
  in `current_framework` — `Sarek_transpile.set_framework` wrote that
  module-level ref for CUDA/OpenCL/Metal/WGSL, so after any earlier transpile a
  later direct `generate` on a `[%native]` kernel injected with the leaked tag
  instead of raising. GLSL and WGSL silently emitted
  `/* native code not supported in <lang> */` and continued, producing a
  shader missing the operation the kernel asked for with no diagnostic. GLSL
  and WGSL now raise the same shared `Sarek_ir_codegen.native_block_refusal`
  the other three do, naming what the caller should do instead (express the
  operation in Sarek, or route it through PTX, which is the one backend that
  still serves `[%native]` directly). Along the way, the per-generation facts
  each backend needs while emitting — the kernel's variant table on every
  backend, plus GLSL's helper names, f64-softmath state and vec-param table,
  and WGSL's scalar-param names — moved from module-level `ref`s (a
  module-level `Hashtbl` in the case of GLSL's vec-param table), shared and
  clobbered across generations including across domains, to a value threaded
  through the emit calls. The
  `current_framework` tag was not threaded; it was removed outright and
  replaced by a per-backend compile-time constant, so there is no longer a
  module-level slot for it to leak through. The sequential half of this — a
  later generation on the same domain reading an earlier one's leftover state —
  is pinned by `framework_tag_is_not_module_state`; the cross-domain half, by
  `generations_do_not_interleave`.

- The CUDA branch of `sarek_f32_barrier` no longer emits
  `asm volatile("" : "+f"(x))`. At an f16 narrowing it contributes zero PTX
  instructions, so `ptxas` receives an identical instruction stream and the
  cubins are byte-identical with and without it — re-measured on CUDA 13.3 for
  sm_75 through sm_121. What keeps the f32 multiply out of the narrowing on
  NVIDIA is `ptxas`, machine-checked by `test_cuda_f16_sass`. NOTE the same
  barrier is *not* inert at a `mul`→`add` site (PTX `mul.f32`+`add.f32` instead
  of `fma.rn.f32`), but `ptxas -O1`+ re-contracts that under the default
  `-fmad=true`, so it is still not a usable contraction barrier on NVIDIA — use
  `Sarek_df64`'s `mul_rn`. The AMDGPU `"+v"` barrier, which IS load-bearing, is
  unchanged. Removing a no-op that read as protection; no behaviour change.

### Fixed

- A shared-memory array of a record type behaved four different ways across
  nine devices, none of them right. `let%shared (s : tri) = 4l` followed by
  `s.(i).a <- e`, measured on this host: the Interpreter ×2 raised
  `assignment target of .a (got unit)`; Native accepted the store and wrote
  EVERY slot (`7 7 7 7`, want `7 0 0 0`); CUDA/PTX ×2 raised
  `unsupported construct: btype of custom type`, naming neither the array nor
  the type; and OpenCL ×2 and Vulkan ×2 failed inside the DEVICE compiler,
  OpenCL with `unknown type name 'Test_..._tri'` and Vulkan with a glslang
  syntax error at the shared declaration. Three separate causes. (1) Native's
  `alloc_shared_with_key` filled the array with `Array.make size default`, so
  every slot of a boxed element type was the same allocation — it now takes a
  per-slot thunk and calls `Array.init`, and the identical `Array.make` in the
  `create_array n Local` path is fixed with it. (2) The interpreter mapped a
  record element type to `VUnit`, so there was nothing to store into — it now
  builds a zeroed `VRecord` — or, for a variant, a `VVariant` carrying the
  first NULLARY constructor, the same one Native's `default_value_for_type`
  picks, falling back to the first constructor with zeroed payloads where
  there is no nullary one (a case in which Native has no default at all and
  raises, so the interpreter is more defined there rather than agreeing) — per slot in
  `Sarek_ir_interp_value.alloc_kernel_array`. The tag is
  `Hashtbl.hash ctor mod 256`, the encoding `EVariant` and the two matchers
  already use and which is now the single `variant_tag_of_ctor`; a positional
  index there produced a `VVariant` no arm could match, so reading a default
  variant slot raised `Pattern match failure in SMatch` on the Interpreter
  while Native answered the nullary constructor. That replaces three identical
  copies of the old init table in `Sarek_ir_interp_eval` plus a fourth, NARROWER
  one on `Sarek_ir_interp`'s `DShared` kernel-parameter path, which mapped only
  `TInt32` and `TFloat32` and everything else to `VUnit`. Unifying them
  therefore also widens that path: `TInt64`, `TFloat64`, `TBool`, `TFloat16` and
  `TUint8` go from `VUnit` to their typed zeros there. That is a fix, but it is
  a behaviour change outside backlog-206's shape and it is stated rather than
  folded into the word "copies". (3) OpenCL and Vulkan had no shared-memory gap at all:
  `register_types_from_typ` ran over PARAMETER types only, so a record named
  nowhere but the shared declaration was never emitted as a `struct`; it is now
  top-level and runs at both kernel-array declaration sites. The tell was that
  the same kernel written with a whole-slot store (`s.(tid) <- {...}`) compiled
  and ran correctly on all seven non-PTX devices throughout, because the record
  literal registered the type — so a whole-slot-store test would have been green
  the whole time, and `test_shared_record_slots.ml` exercises both shapes.
  CUDA/PTX still refuses, and the test asserts the refusal rather than
  tolerating it: PTX has no struct type and
  `Sarek_ir_ptx_mem.emit_agg_elem_addr` refuses state-space aggregates on
  purpose. What changed there is that the message names the array, the element
  type and the state space. Metal, HIP and WGSL were not measured — no such
  device on this machine — and nothing is claimed about them.

- A `[@@sarek.type]` record with a record-typed field had its inner struct
  emitted AFTER the struct that referenced it, so no kernel touching a nested
  record compiled on any shader backend. Both declaration-emission loops walked
  `kern_types` in list order, and that is not a dependency order: the PPX
  prepends the types reachable through the registry to the ones the kernel
  payload declares. Measured on an RX 7900 XTX host, a read-only nested access
  (`dst.(tid) <- src.(tid).mid.b`) failed on OpenCL ×2 with
  `unknown type name 'Test_..._triple'` and on Vulkan ×2 with a glslang parse
  error at the field line; a three-level chain and two independent nested types
  failed the same way. `Sarek_ir_codegen.sort_record_types_by_dependency` now
  orders the declarations, stably — ties break on the incoming list position,
  so an already-correct order is returned unchanged and no committed golden
  moved. A cycle between distinct record declarations raises `Record_type_cycle`
  instead of being emitted in some wrong order (a self-referencing field is
  dropped, not reported); that path is unreachable through the PPX, which
  refuses a self- or mutually-referencing record field while resolving its
  alignment (measured: both spellings stop at *unknown alignment for field
  type*), and is kept as a backstop for hand-built IR. Interpreter and Native
  were never affected — they carry values, not struct declarations. This orders
  records against records only; the two cross-kind halves it left live are the
  entry below, which also replaced the two identifiers named here — see the
  BREAKING note under Changed for what they became.
- A dependency edge CROSSING between a record declaration and a variant
  declaration was ordered by nothing, because each backend family sorted one
  kind inside its own emission loop and the families disagreed on which loop ran
  first. Both halves are reachable from ordinary `[@@sarek.type]` source and both
  were measured on an RX 7900 XTX / Ryzen 7950X host, two OpenCL devices and two
  Vulkan devices. A variant with a RECORD payload is red on both OpenCL devices
  (`error: unknown type name 'Test_record_variant_decl_order_probe_pt'` at the
  union member) and green on both Vulkan devices; a record with a VARIANT-typed
  field is the exact inverse — red on both Vulkan devices
  (`syntax error, unexpected IDENTIFIER` at the field line) and green on both
  OpenCL devices. Each family being green on the shape the other fails on is why
  a reproducer built on either half alone reports the ordering fixed. Reaching
  either half needs a RUNTIME-SELECTED constructor: static tag erasure (L14
  S1/S2) reduces a variant-typed local or record field written by a literal
  constructor application, and an erased variant never reaches `kern_variants`,
  so a literal-constructor reproducer is green on every device while covering
  nothing. All five generators that declare struct types (CUDA — also HIP —
  OpenCL, Metal, GLSL, WGSL) now emit records and variants from ONE interleaved
  topological pass over both lists together, `Sarek_ir_codegen.gen_type_decls`,
  instead of two per-kind loops. The tie-break is still the incoming index and
  each backend passes a `~tie_break` naming the order its own two loops used to
  run, so an edge-free kernel's emitted source is byte-identical and no committed
  golden moved. Node identity in the sort is the POSITION in the list, not the
  type name: a name-keyed self-edge drop discards a record's genuine edge to a
  SAME-NAMED variant, which `mangle_name` and fusion can both produce, and that
  is precisely the edge class this entry is about. The cyclic-value guard stays
  keyed on PHYSICAL node identity, so a cyclic `elttype` value still terminates
  whatever closes the cycle — including `let rec t = TVec t`, which has neither a
  record nor a variant on it. What this does NOT fix: a type referenced but never
  declared. The PPX registers a variant in `kern_variants` without registering a
  record that appears only in its payload, and no ordering pass can supply a
  declaration that is absent — it still surfaces as the backend's own *unknown
  type name*. Interpreter, Native and PTX were never affected: the first two
  carry values, and the PTX emitter declares no struct types.

- `v.(i).f <- e` — an in-place record-field store on a vector element — did
  something different on each CPU backend, while working on CUDA/PTX. The
  Interpreter **refused** it (`Unsupported_operation "record field assignment"
  / "not fully supported"`). Native had two failure modes split by mutability:
  on a **mutable** field it accepted the store and **silently dropped** it — the
  generated OCaml was a setfield on the fresh record `Vector.get` had just
  marshalled out of storage, so the write hit a temporary, nothing raised, and
  the vector kept its old values — while on an **immutable** field the same
  setfield did not compile (`The record field b is not mutable`), loud but
  misdiagnosed, which is what pushed users to add `mutable` and so into the
  silent half. Both halves are pinned.

  The nested form `v.(i).f.g <- e` needed a second fix on top of the depth-1
  one, on both CPU backends: with depth 1 fixed the Interpreter still read the
  intermediate record through the registry's copying `get_field` and Native still
  matched only the depth-1 shape, so the chained store was silently dropped by
  both. (Before either fix the Interpreter refused every field store, nested or
  not; the silent nested drop is the state the depth-1 fix alone would have
  shipped.) Asserted on every enumerated device, reading the values back and
  checking the neighbouring fields at both levels, because a dropped store is
  otherwise indistinguishable from a kernel that never ran.

  Two limits, stated rather than implied. On Native the store is now a
  whole-element read-modify-write, which makes two threads writing *different*
  fields of the *same* element a lost-update **race** there — an interleaving
  that reads both elements before either writes back loses one store, a
  serialised one loses neither — where the C-family, PTX and interpreter paths
  touch disjoint locations and cannot interfere. Read off the expansion, not
  observed, and covered by no test (backlog-207). And a record type declared
  *inside* the kernel payload is read through a generated getter that has no
  setter, so a store to one is now **refused with a message** instead of silently
  emitting a setfield.

- A record bound to a `let` inside a kernel was a live window onto vector
  storage in the Interpreter rather than a copy, so mutating the local mutated
  the vector element it came from. Detaching it copies through nested records and
  variant payloads, and refuses a chain deeper than 64 links. What that guard
  replaces is not a hang: the recursion is not tail-recursive, so an unguarded
  cyclic value dies with `Stack_overflow` — an untyped crash naming neither the
  value nor the binding — after tens of seconds of allocation and GC thrash
  (measured on one host; the seconds are a property of that `ulimit -s`, so the
  direction is the claim, not the figure). The depth-65 refusal itself is read
  off the marshaller and `bind_var`, not observed, and no committed test covers a
  65-link chain.

- `Sarek_df64` silently ran at plain float32 precision on real NVIDIA
  hardware (CUDA/PTX and NVIDIA OpenCL): `ptxas` contracted the multiply in
  `two_prod` into the `add`/`sub` of the `quick_two_sum` closing `df64_mul`,
  rebuilding the exact product and cancelling the TwoProd error term.
  `mul`/`div`/`sqrt` degraded from ~2^-47 to ~2^-24 with no error reported.
  `two_prod` now forms its product with `fma a b 0.0`, which cannot be fused
  again. Measured on a GTX 1070 (sm_61, CUDA 12.9): mul 5.92e-08 → 9.07e-15,
  div 5.64e-08 → 5.08e-15, throughput unchanged. A CPU-only regression guard
  (`test_df64_no_contraction`) asserts the emitted PTX contains no
  contractable `mul.f32`. The df64 per-backend precision table now names the
  device and toolchain behind every figure — the previous table generalised
  AMD-only measurements to "CUDA/PTX", which is why this went unseen.
- df64 precision gates could not distinguish a working df64 from a collapsed
  one. `test_df64`, `test_real64` and `test_real64_single_source` all widened
  their tolerance to `0x1p-22` (2.38e-07 — four times the float32 unit
  roundoff) on the backends with a documented deviation, so a df64 that had
  degraded all the way to plain float32 (measured 5.84e-08 on RADV) and one
  meeting its contract (9.07e-15) both read PASS. The widening also keyed on
  the bare `Vulkan` framework tag, sweeping in NVIDIA Vulkan, which meets the
  full contract. All three tests now hold every device to the derived contract
  bound (2^-47 add/sub, 2^-46 mul/div/sqrt) and express the documented
  deviations as an explicit expected-failure band — upper end 2^-23, twice the
  float32 unit roundoff — keyed on driver identity (Mesa RADV, Mesa ANV) in
  `Test_helpers.df64_known_deviation`. Degrading past plain float32 now FAILs
  even on an allowlisted device, and so does an allowlisted device that starts
  MEETING the contract (strict XPASS, as in pytest's `xfail(strict=True)`) —
  the run goes red naming the match arm to delete, so the allowlist cannot rot
  behind a green exit code. Red-proved twice: replacing `df64_mul` with a plain
  float32 multiply (old gate PASS on both RADV devices and Native at 1.43e-07
  vs tol 2.38e-07, new gate FAIL), and adding a bogus allowlist entry for an op
  that already meets the contract (all three tests exit 1 with the stale-entry
  message).
- PTX backend emits `sqrt.rn.f32` rather than `sqrt.approx.f32` for the float32
  `sqrt` intrinsic — the same correctness fix already applied to division
  (`div.approx.f32` → `div.rn.f32`, audit finding M2), which had missed `sqrt`.
  Dumping the generated PTX showed `sqrt.approx.f32` (~1 ulp) was the only
  non-correctly-rounded instruction in the whole `df64_sqrt` body, where it
  serves as the Newton seed. This is a global change: every f32 sqrt in every
  PTX kernel is now correctly rounded. Measured worst-case relative error over
  each test's own input set, on a GTX 1070 Max-Q (sm_61, CUDA 12.9, driver
  580.119.02): `df64_sqrt` 1.42e-14 (failing) → 8.53e-15 in `test_df64`, and
  1.68e-14 (failing) → 8.87e-15 in `test_real64`'s df64 fallback. Both post-fix
  figures coincide with the interpreter's for the same inputs — agreement
  between summary statistics, not element-wise identity — so the seed accounted
  for the whole gap these tests can see. Sampled maxima on one device and
  toolchain, not bounds. Costs ~12% kernel time on a sqrt-dominated benchmark
  (`bench_nbody` n=4096: 1.535 ms → 1.722 ms); `rsqrt` is unchanged for code
  that wants the fast form. The same bug class remains open in the OpenCL
  backend (no `-cl-fp32-correctly-rounded-divide-sqrt`, sqrt 1.81e-14) and the
  Vulkan backend (1.68e-14 on NVIDIA); see the `KNOWN RESIDUAL` block in
  `sarek/Sarek_df64/Sarek_df64.ml`.
- Indexed kernel-argument container with strict launch validation, honored
  across all six backends (out-of-order/sparse `set_arg` now correct)
- Unambiguous, collision-resistant compile-cache keys (kernel name included,
  digest-per-field); CUDA kernel-cache eviction on context destroy
- e2e test alias actually executes tests; vacuous tests now assert real
  results; negative suite runs in CI
- Vulkan push-constant scalar binding by logical index; 64-bit write
  alignment; buffer-index validation
- Benchmark dashboard HTML escaping at all sinks (stored-XSS hardening)
- CPU data kept authoritative after Interpreter runs

## 2026-06

### Added

- CUDA backend split into CUDA/PTX and CUDA/C devices (PTX is the default)
- `.shared` (DShared) emission in the PTX backend with formal spec
  (`specs/ptx-dshared-formal.md`)
- T3-SEMANTIC theorem work for the convergence-safety and type-safety
  formal projects
- WGSL/WebGPU codegen backend, in-browser playground and Learn course

### Changed

- Breaking: removed first-party `Obj` escape hatches from Sarek/SPOC execution
  paths.
  - Kernel vector arguments now cross framework/plugin boundaries through typed
    existential accessors and runtime type witnesses.
  - Custom value conversion now uses typed helper lookup instead of raw runtime
    representation casts.
  - Native and interpreter plugin buffer copies use typed Bigarray operations.
  - Custom shared-memory arrays are keyed by typed witnesses to reject
    mismatched reuse.
  - Legacy native direct execution now accepts typed `Framework_sig.exec_arg`
    arrays.

## 2026-01

### Changed

- Documentation cleanup and modernization
  - Comprehensive README rewrite with clear SPOC/Sarek distinction
  - Added sarek/ directory navigation guide
  - Created CONTRIBUTING.md with project guidelines
  - Removed pretentious language ("Grade A", "100%", etc.)
  - Added AI assistance acknowledgment
- Code quality improvements
  - Eliminated 49 failwith calls from Native and Interpreter plugins
  - Added structured error handling using Backend_error pattern
  - Created test suites for Native and Interpreter plugins
  - Added READMEs for Native, Interpreter, ppx_intrinsic, Sarek_float64
- CI/CD modernization
  - Added unit test execution (dune runtest)
  - Created fast benchmark suite (~20s) for CI
  - Integrated coverage measurement with bisect_ppx
  - Simplified workflow to single build job
- Repository cleanup
  - Removed AI artifact files (AGENTS.md, etc.) from history
  - Removed tracked build artifacts (*.exe, *.log, etc.)
  - Updated .gitignore for better hygiene

## 2025-12

### Added

- Full PPX rewrite replacing Camlp4
  - PPX-based kernel syntax ([%kernel ...], [%ktype ...])
  - Type registry for cross-module GPU types
  - Intrinsic registration system (Sarek_stdlib)
  - Float32 stdlib module with math intrinsics
  - Pragma support for loop unrolling hints
- Kernel fusion system (Sarek_fusion.ml)
  - Vertical fusion for producer-consumer patterns
  - Reduction fusion (map + reduce)
  - Stencil fusion with radius tracking
  - Auto-fusion heuristics
- Clean intermediate representation (Sarek_ir.ml)
- BSP superstep syntax with barrier synchronization
- Warp convergence tracking (Sarek_convergence.ml)
- Core primitives with convergence information
- Comprehensive test suite (unit, negative, e2e tests)

### Changed

- Removed Camlp4 dependency entirely
- Improved type inference for kernel parameters
- Better compile-time error messages

## 2024-2025

### Changed

- GPU Backend Overhaul (all 4 backends: CUDA, OpenCL, Vulkan, Metal)
  - Eliminated all failwith calls (94 total removed)
  - Implemented Backend_error.Make functor pattern for structured errors
  - Refactored code generators for maintainability
  - Added comprehensive unit test suites (19-20 tests per backend)
  - Created professional documentation (9-26KB per backend)
  - Improved error messages with context
- OCaml 5.x migration
  - Updated to OCaml 5.4.0
  - Migrated from Domains to Effect handlers where appropriate
  - Maintained backward compatibility

## 20210823

- Add PPX extension to declare external GPGPU kernels
- Update Samples to use PPX instead of Camlp4 extension
- Update dune/opam files for opam release

## 20210816

### Added

- Build with dune
- Compatible with OCaml 4.12
- Switch to github actions for CI
