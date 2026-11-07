# A typed optimisation policy for SPOC/Sarek and a downstream consumer

**Status.** M1 and M2 implemented (`spoc/ir/Sarek_policy.ml`); M3 planned. Per
milestone:

| Milestone | Status | Content |
|---|---|---|
| M1 | implemented | `Sarek_policy`: one typed store for every precision grant and consent; the three coopmat opt-ins and `Sarek_f16_policy.accept_single_rounding` are wrappers over it; no behaviour change (golden `sarek/tests/codegen_golden/test_policy_golden.ml`) |
| M2 | implemented | window, epilogue, matrix, narrowing and unroll decide through `Sarek_policy` from the effective policy of their kernel (grants at process and kernel scope, meet over fused members; process restrictions); the source memo is keyed on the policy; epilogue fusion on CUDA/OpenCL/Vulkan under `Cross_kernel_contraction` (details in 3.1a) |
| M3 | planned | `SAREK_OPT_DISABLE`, `SAREK_EXPLAIN`; the replaced switches are removed outright; `gh-pages/docs/environment.md` completed |

**Date:** 2026-10-11, revised 2026-10-12 for the owner decisions. **Sources:**
SPOC `origin/main` `5bcb3a8e` (table 1.1), a downstream LLM inference engine's
configuration at the same date (table 1.2).

SPOC and the downstream engine have about 50 optimisation, precision and resource switches,
each with its own scope, parser and reporting. This note describes one typed
policy that SPOC's compiler and launch path query, one explain report saying
what was applied and why, and one downstream precision setting. Owner rules kept:
exact optimisations on by default, deviations and resource contracts explicit,
no `Obj`. A switch the policy replaces is removed, not aliased (Q6). Section 2
lists six points where the evidence changes the discussed proposal, chiefly: the
family depends on the backend; exact is not always faster; one numeric bound
cannot express the opt-ins; resource contracts are handles, not settings.

## 1. Inventory

Families: **S** semantics-preserving, **P** precision-changing, **C** consent
to rewrite the program's own code, **R** resource contract, **T** tuning
(results unchanged), **U** unsafe override. Out of scope: selection,
diagnostic and test variables (`SPOC_DISABLE_*`, `SAREK_DEBUG`,
`SPOC_GPU_STATS`, `SPOC_BENCH_*`, the engine's verbosity variable, ...).

### 1.1 SPOC/Sarek (`origin/main` 5bcb3a8e, 2026-10-12)

Rows marked "changed" differ from the first version of this note (3ad881b0),
rows marked "new" were not in it.

Rows 1, 5, 7, 8, 9b, 15 and 16 describe switches that M3 removed (section 3.4); the
inventory records what existed at the audited revision.

| # | Name | Kind | Fam. | Default | Set by | Scope | Ref |
|---|---|---|---|---|---|---|---|
| 1 (changed) | `SAREK_LAUNCH_WINDOW` (off: exactly `0/false/off/no`) | env | S | on; default backends HIP, OpenCL, Vulkan, CUDA/PTX, CUDA/C, Metal; never CPU devices | user | process, read at init | `sarek/execute/Launch_window.ml:159-172`; `sarek/execute/Execute.ml:1218-1220` |
| 2 | `Execute.Window.set_enabled` | API | S | env value | program (downstream) | process | `sarek/execute/Execute.ml:1227`; `sarek/execute/Launch_window.ml:488-490` |
| 3 (changed) | `Execute.Window.set_backend_enabled` | API | S | 6 backends | program | per backend | `sarek/execute/Execute.ml:1231-1232`; `sarek/execute/Launch_window.ml (set_backend_enabled)` |
| 4 | `Launch_window.max_groups` / `max_members` (8/8) | API | T | 8/8 | program | process | `sarek/execute/Launch_window.ml:153`, `sarek/execute/Launch_window.ml:155` |
| 4b (new) | `Launch_limits.set_arg_slot_limit` / `set_push_constant_limit` | API | T | per backend | program/tests | per backend | `sarek/execute/Launch_limits.ml:25`, `sarek/execute/Launch_limits.ml:53` |
| 5 (changed) | `SAREK_LAUNCH_EPILOGUE` (off only) | env | S on HIP, Metal; P elsewhere | HIP + Metal | user | process, init | `sarek/execute/Launch_epilogue.ml:46-53` |
| 6 (changed) | `Execute.Window.set_epilogue_backend` -> `Launch_epilogue.set_backend_enabled` | API | S/P | HIP + Metal | program | per backend | `sarek/execute/Execute.ml:1241-1243`; `sarek/execute/Launch_epilogue.ml:66-73` |
| 6b (new) | `Fp_exactness.mark_inexact ~backend ~reason` (Metal without conformant compile options) | API (plugin) | restriction | none | Metal plugin | per backend, one-way | `sarek/framework/Fp_exactness.ml:17-35`; `sarek-metal/Metal_api.ml:270-280`, `sarek-metal/Metal_api.ml:744-748` |
| 7 (changed) | `SAREK_MATRIX_REWRITE` off/exact/auto (any case; unknown warns, then auto) | env | S (int32) + gate on C | auto | user | process, init | `sarek/execute/Matrix_specialise.ml:29-52` |
| 8 | `Matrix_specialise.set_mode` / `mode` (= `Sarek_matrix.set_mode`) | API | same | Auto | program | process | `sarek/execute/Matrix_specialise.ml:54-56`; `sarek/Sarek_matrix/Sarek_matrix.ml:25-27` |
| 9 (changed) | `Matrix_specialise.choose ?precision ?explicit ?stream` (only `run_vectors*`) | API | P | f16 then bf16 | library | per launch | `sarek/execute/Matrix_specialise.ml:409-425`; `sarek/execute/Execute.ml:1929-1956` |
| 9b (new) | `Matrix_specialise.set_reporter` | API | report | none | program | process | `sarek/execute/Matrix_specialise.ml:385-387` |
| 10 | `Sarek_coopmat.accept_relaxed_accumulation` | API | P | no | program | process, one-way | `spoc/ir/Sarek_coopmat.ml` (wrapper of `Sarek_policy.grant`) |
| 11 | `accept_operand_rounding ~precision` (f16, bf16, bf16x3) | API | P | no | program | process, one-way, per precision | `spoc/ir/Sarek_coopmat.ml` (wrapper) |
| 12 | `accept_automatic_float_rewrite` | API | C | no | program | process, one-way | `spoc/ir/Sarek_coopmat.ml` (wrapper) |
| 13 (changed) | narrowing-barrier removal when proven (merged #480), CUDA-C generator only (HIP, CUDA/C) | built-in | S | on | nobody | per site | `sarek/codegen/Sarek_ir_cuda.ml:61-97` |
| 14 (changed) | `Sarek_f16_policy.accept_single_rounding` (merged) | API | P | no | program | `All_kernels` or `Kernel n`, one-way | `spoc/ir/Sarek_f16_policy.ml` (wrapper) |
| 15 (changed) | `SAREK_AUTO_UNROLL=0` (exactly `0`), read at every annotate; CUDA-C, OpenCL, Metal, GLSL generators | env | S | on | user | process | `sarek/codegen/Sarek_ir_unroll.ml:17-18`, `sarek/codegen/Sarek_ir_unroll.ml:121-124` |
| 16 | `SAREK_MATRIX_REPORT=1` | env | report | off | user | process | `sarek/execute/Matrix_specialise.ml:389-402` |
| 17 | `SAREK_DOMAIN_COUNT` | env | T | recommended | user | pool creation | `sarek/interp/Cpu_pool_lifecycle.ml:52` |
| 18 (changed) | `SAREK_CPU_POOL_MAX_DOMAINS`, `SAREK_CPU_POOL_IDLE_MS` (now documented) | env/API | T | none / 1000 ms | user/program | process | `sarek/interp/Cpu_pool_lifecycle.ml:23-46` |
| 19 | `SPOC_METAL_BATCH` | env | T | 32 | user | process | `sarek-metal/Metal_api.ml:362` |
| 19b (new) | `SPOC_VULKAN_BATCH` (default 32 again after #504's interim default 1; `1` is the fallback; not an `optimisation` of the explain report, so `SAREK_EXPLAIN` does not mention it) | env | T | 32 | user | process | `sarek-vulkan/Vulkan_api_batch.ml:23` |
| 19c (new) | `SPOC_VULKAN_FENCE_TIMEOUT_S` | env | T (diag) | unbounded | user | process | `sarek-vulkan/Vulkan_api_base.ml:64` |
| 20 (changed) | `SAREK_HIP_IGPU_UNCACHED=0` (now documented) | env | U | uncached | user | process | `sarek-hip/Hip_api.ml:279` |
| 20b (new) | `SAREK_DISK_CACHE=0` (OpenCL #489, Vulkan SPIR-V) | env | T | on | user | per lookup | `sarek/framework/Framework_cache.ml:73` |
| 21 (changed) | `Transfer.alloc_pinned` / `register_host` / `free_pinned` (now also Vulkan) | API | R | none | program | per buffer | `sarek/core/Transfer.ml:1558`, `sarek/core/Transfer.ml:1603`, `sarek/core/Transfer.ml:1653` |
| 22 | `Advanced.Pinned` | API | R | none | program | per buffer | `sarek/core/Advanced.ml:53`, `sarek/core/Advanced.ml:107` |
| 23 | `Host_ptr_buffer.of_bigarray` / `attach` | API | R | copies | program | per buffer | `sarek/core/Host_ptr_buffer.ml:121`, `sarek/core/Host_ptr_buffer.ml:133` |
| 24 | `?stream`, `Transfer.create_stream` | API | R (async) | synchronous | program | per copy | `sarek/core/Transfer.ml:949` |
| 25 | `Advanced.Unified` (placeholder) | API | R | none | none | none | `sarek/core/Advanced.ml:19` |

Decision points and caches (at 5bcb3a8e):

- **Window.** `Execute.run_jit` (`sarek/execute/Execute.ml:1469`) calls
  `defers_on` (`sarek/execute/Execute.ml:1347`; removed in M3, replaced by
  `Launch_policy.deferred` over `Window_gate.refusal`), then
  `Window.defers_device` (`sarek/execute/Execute.ml:1218-1220`), then
  `Launch_window.defers` (`sarek/execute/Launch_window.ml:177-178`). The fused-kernel cache is keyed by
  the members' `ir_id`s and content-named
  (`sarek/execute/Launch_window.ml:300-328`).
- **Epilogue.** `Launch_window.submit` reads `w.epilogue`
  (`sarek/execute/Launch_window.ml:420`), installed by `Launch_epilogue.install`
  (`sarek/execute/Launch_epilogue.ml:455-460`). The join checks
  `enabled c.backend` (`sarek/execute/Launch_epilogue.ml:433-434`);
  `enabled b = List.mem b backends && Fp_exactness.is_exact b`
  (`sarek/execute/Launch_epilogue.ml:58-60`). Its fused, coverage and shape
  caches have no policy in their keys.
- **Matrix.** `Matrix_specialise.decide` (`sarek/execute/Matrix_specialise.ml:335-381`)
  reads the mode, then `lowering_for` (`sarek/execute/Matrix_specialise.ml:238-253`),
  then `Matrix_admission.device_admits`
  (`sarek/execute/Matrix_admission.ml:163-177`), then
  `Sarek_coopmat.policy_verdict`. The per-IR plan is memoised; grants are
  re-checked at each launch.
- **Narrowing proof.** The CUDA-C generator reads
  `Sarek_f16_policy.single_rounding_reason ~kernel:kern_name`
  (`sarek/codegen/Sarek_ir_cuda.ml:71-81`). The source memo
  (`sarek/execute/Execute.ml:1006-1020`) was keyed by IR identity, generator,
  block and SoA parameters, with no policy in the key: a kernel generated before
  a grant kept its barriers (fixed in M2: the entries carry a `Policy_stamp`).
- **Unroll.** `Sarek_ir_unroll.enabled ()` at every annotate
  (`sarek/codegen/Sarek_ir_cuda.ml:1049`, `sarek/codegen/Sarek_ir_metal.ml:1208`,
  `sarek/codegen/Sarek_ir_opencl.ml:1179`, `sarek/codegen/Sarek_ir_glsl.ml:2169`).

### 1.2 Downstream engine (its configuration document)

This table is the downstream engine's, at the date above; it is not maintained here.
Variable names are the engine's and are given here by role.

| # | Name | Kind | Fam. | Default | Scope |
|---|---|---|---|---|---|
| 26 | prefill-MMA switch -> `Mma_gemm.enable` -> grant #10 | env | P (+ the engine's own f16 rounding of weights and activations) | off | process |
| 27 | `Prefill.without_launch_window` (window off during prompt chunks unless `SAREK_LAUNCH_WINDOW` is set; that variable is removed in M3, so the engine moves to `SAREK_OPT_DISABLE=window` or its own check) | internal | T over S | on | per prompt chunk |
| 28 | `Staging` buffer via `Transfer.alloc_pinned` (64 MiB) | internal | R | **always on** | per device |
| 29 | RAM-register switch -> `Transfer.register_host` on the L2 pool | env | R | off | process |
| 30 | load mode `mmap` | env | R (host memory) | direct reads | process |
| 31 | RAM tier `pagecache`, exclusive RAM tier, prefetch switches | env | T (tier strategy) | off | process |
| 32 | expert-cache size, no-tuned switch, RAM-cache size, RAM preload, mirrors, mirror weights, mirror stripe size, prefill chunk size (8 variables) | env | T (sizes) | see doc | process |
| 33 | device selection variable (`opencl:*` opt-in loads the plugin) | env | selection | first GPU | process |
| 34 | CPU-experts and CPU-threads variables (plan only, the engine's CPU expert executor plan of 2026-10-11) | env | R + T | none | process |

**Inventory size:** 34 rows (25 SPOC, 9 downstream), about 50 named variables and
calls. Precision or consent decisions sit in ten rows (5-12, 14, 26) across
five modules (`Launch_epilogue`, `Matrix_specialise`, `Sarek_coopmat`,
`Sarek_f16_policy`, the engine's `Mma_gemm`), each with its own scope and reporting.
After M1 the grants of rows 10-12 and 14 share one store.

**Drift (flagged, not fixed):** the engine's `Bench_report.env` keeps only its
own variables (the engine's bench report module), so a stored
report cannot show `SAREK_OPT_DISABLE=window`.

## 2. Where the evidence contradicts the discussed proposal

**E-a. The family depends on the backend.** Epilogue fusion is off outside HIP
and Metal because there "the device compiler may contract a multiply of the
producer with an add of the consumer [...], which changes the result by an ulp"
(`sarek/execute/Launch_epilogue.ml`). Classification is per (optimisation,
backend).

**E-b. Exact is not always faster.** The launch window gives 35B decode 112
vs 101 tok/s (downstream measurement) but cuts a cold 0.8B first prompt from 703 to
133 tok/s, so the engine turns it off around prompt chunks (row 27).
"Only a diagnostic off-switch" would remove that lever. Owner decision Q4: SPOC
decides the prompt-phase case itself (fused-kernel disk cache, cost model);
`Window.set_enabled` stays until then.

**E-c. One numeric bound cannot express the opt-ins.** The accumulation bound
is derived per configuration and K; operand rounding has three precisions
(bf16x3 near-f32); single rounding is "at most one binary16 ulp plus
|a·b|·2^-24" and more accurate than the interpreter (#480). Different kinds,
not one scale: the budget is a set of named deviations, no numeric bound (Q2).

**E-d. #440 split consent from accuracy** ("granting one never enables more
than it names"). The store keeps that: `Deviation` and `Consent` are separate
atoms and no grant implies another (checked over all 7 x 7 pairs by
`spoc/ir/test/test_sarek_policy.ml`). A profile may list consent next to
accuracy atoms (Q1), each atom reported.

**E-e. Precision grants have no env variable on purpose** (#440: "a statement
in the program's source"). So SPOC env vars may only restrict (Q3); the engine may
turn its own setting into grants, being then the one making the statement.

**E-f. Resource contracts are ownership, not code generation.** `Pinned.t`,
`Host_ptr_buffer` and streams carry lifetime and sync obligations
(`publish_host_writes`, release refused while copies are in flight); SPOC
cannot apply them on the caller's behalf. They stay typed handles and only
join the report; a memory setting, if any, belongs to the engine.

## 3. Design

### 3.1 Types (implemented in M1: `spoc/ir/Sarek_policy.mli`)

```ocaml
type operand_precision = Operands_f16 | Operands_bf16 | Operands_bf16x3
type deviation =
  | Accumulation_order                  (* accept_relaxed_accumulation *)
  | Operand_rounding of operand_precision
  | Single_rounding                     (* Sarek_f16_policy *)
  | Cross_kernel_contraction            (* epilogue fusion; read from M2 *)
type consent = Automatic_float_rewrite  (* #440 *)
type grant = Deviation of deviation | Consent of consent
type optimisation =                     (* every switchable transformation *)
  | Launch_window | Epilogue_fusion | Matrix_rewrite | Narrowing_proof
  | Auto_unroll
type scope = Process | Kernel of string      (* refuses "sarek_kern" *)
type refusal = Blank_reason | Default_kernel_name
type evidence = Proof of string | Granted of string | Exact_by_construction
type decision = Applied of evidence | Refused of string

module type BUDGET = sig
  type t                               (* abstract: grants with reasons *)
  val exact : t                        (* bottom: no grant *)
  val add : grant -> reason:string -> t -> (t, refusal) result
  val permits : t -> grant -> bool
  val reason : t -> grant -> string option
  val meet : t -> t -> t               (* used for fused kernels *)
  val leq : t -> t -> bool
  val grants : t -> (grant * string) list
  val digest : t -> string             (* grant set only; for cache keys *)
end
module Budget : BUDGET

val grant : scope -> grant -> reason:string -> (unit, refusal) result
val reason : ?kernel:string -> grant -> string option
val budget : kernels:string list -> Budget.t
val refusal_message : refusal -> string
```

`Sarek_policy` lives in `sarek_ir`, the lowest layer (no dependencies), so code
generation, the capability gate and the launch path can all ask it. The grants
of rows 10-12 and 14 are thin wrappers: same refusals, same messages. In M1 the
`optimisation`, `evidence` and `decision` types and `Cross_kernel_contraction`
are vocabulary for M2 and M3; only the narrowing reader consults a kernel scope
(decision points that know the kernel consult its scope; from M2 all do).

### 3.1a The effective policy (implemented in M2)

M2 adds how each optimisation declares what it needs and how it decides. The
module-type sketch of the first draft (one module per optimisation) became two
functions and a record, because the facts of each decision point differ and stay
with that point:

```ocaml
type effective = { budget : Budget.t; disabled : optimisation list }
val effective : kernel:string -> effective
val effective_for : kernels:string list -> effective      (* meet over kernels *)
val requires : optimisation -> backend:string -> grant list option
                                   (* Some [] = exact there; None = not offered *)
val decide : effective -> optimisation -> backend:string -> decision
val register_fused : name:string -> members:string list -> unit
val restrict : optimisation -> unit   (* one-way, process-wide *)
val restricted : optimisation -> bool
val version : unit -> int             (* bumped by every change of the policy *)
val set_reporter : (event -> unit) option -> unit   (* Decision record | Ignored_grant *)
val report : record -> unit           (* once per new outcome per (kernel, device, optimisation) *)
```

`requires` table:

| Optimisation | HIP | Metal | CUDA/PTX, CUDA/C, CUDA, OpenCL, Vulkan | Native, Interpreter, WebGPU |
|---|---|---|---|---|
| `Epilogue_fusion` | `[]` | `[]` (while `Fp_exactness.is_exact "Metal"`) | `[Deviation Cross_kernel_contraction]` | not offered |
| `Launch_window`, `Narrowing_proof`, `Auto_unroll`, `Matrix_rewrite` | `[]` | `[]` | `[]` | `[]` |

(HIP: `-ffp-contract=off`; Metal: the pragma, #505; E-a, Q7. `Matrix_rewrite`
needs nothing here: its float lowerings check their own grants, unchanged, now
against the kernel's budget.) `decide` answers, in this order: `Refused
"disabled (SAREK_OPT_DISABLE)"` if the optimisation is restricted; `Refused
"not on <backend>"`; `Refused "no grant <name> (kernel or process)"` for the
first missing required grant; `Applied Exact_by_construction` when nothing is
required; `Applied (Granted reasons)` otherwise. A Metal marked inexact
(`Fp_exactness.mark_inexact`, fast-math fallback) is refused by the epilogue
even with the grant: fast math is more than contraction.

Where each decision point reads the policy (milestone M2):

| Decision point | Reads |
|---|---|
| Narrowing barriers (`Sarek_ir_cuda`) | `Single_rounding` from `budget ~kernels:[kern_name]`; the exact-narrowing proof is skipped (every barrier kept) when `Narrowing_proof` is restricted |
| Unroll (`Sarek_ir_unroll`) | off when `Auto_unroll` is restricted (`SAREK_OPT_DISABLE=unroll`) |
| Source memo (`Execute.generate_source_memo`) | `Policy_stamp`: `version ()` plus the digest of the kernel's effective policy, stored with each source |
| Window (`Launch_window.defers`) | off when `Launch_window` is restricted |
| Epilogue (`Epilogue_policy.decide_pair`, `Launch_epilogue`) | `decide (effective_for [P; C]) Epilogue_fusion ~backend`, per submit, after the per-backend switch and `Fp_exactness` |
| Matrix (`Matrix_specialise`, `Matrix_admission`) | `Accumulation_order`, `Operand_rounding p`, `Automatic_float_rewrite` from the kernel's budget; mode Off when `Matrix_rewrite` is restricted |
| Coopmat gate (`Execute.check_device_capabilities`, interpreter gate and evaluator) | `Sarek_coopmat.policy_verdict_for (budget ~kernels:[kern_name])` |

No grant is recorded without a reader: a `Kernel` grant of any of the seven
atoms changes the decision for that kernel and for no other (tests:
`spoc/ir/test/test_sarek_kernel_scope.ml`, `sarek/execute/test/test_policy_kernel_scope.ml`,
`sarek/tests/codegen_golden/test_f16_single_rounding_golden.ml`).

### 3.2 Composition rules

1. **Process grants are one-way**, like every current opt-in: a check that
   passed stays valid.
2. **Kernel grants add** to process grants for that kernel name only.
3. **A fused kernel gets the meet** of its members' budgets
   (`Sarek_policy.budget ~kernels`). A fusion pass registers the member names of
   the kernel it builds (`register_fused`; `Fusion_horizontal` through
   `Launch_window.fused_kernel`, `Fusion_epilogue` through
   `Launch_epilogue.fused_ir`), and the budget of the fused name is the meet
   over the members, recursively. A `Kernel` grant on the fused kernel's own
   generated name is **not** counted: names are content digests or counters, not
   a statement of the program. Registering a name again adds members (the name
   is then covered only by what all of them allow: content-hash names merge only
   identical fusions, whose members then agree; the fallback names of a fusion
   that cannot be hashed come from one process-wide counter and are never
   shared); a cycle contributes
   nothing. A kernel produced by a rewrite that renames it (the matrix rewrite,
   `<name>_sarek_mm`) is registered with `register_derived ~name ~from`: it has
   exactly its source's budget. The static fusion passes of `Sarek_fusion`
   register like the dynamic ones.
4. **Restrictions only remove**, so toggling them never makes an earlier check
   unsound. There is no public scoped-restriction API (Q4): SPOC decides the
   prompt-phase window itself and keeps a diagnostic switch;
   `Execute.Window.set_enabled` stays, process-wide, until it does.
5. **Caches key on the policy.** #480 said "a kernel generated (and memoised)
   before the call keeps its barriers"; that is no longer true. The source memo
   stores, with each source, the policy `version ()` read before generation and
   the digest of the kernel's effective policy (`Budget.digest` plus the
   disabled optimisations). A launch compares `version ()` (one atomic read,
   no allocation) and, only if it moved, the digest: a grant for another kernel
   keeps the source, a grant for this one regenerates it. The version is read
   before the policy and bumped after a change, so a race can only cause a
   recomputation. Fused-kernel caches (`fused_cache`, the window's `cache`) keep
   the built kernel; the epilogue decision is taken per submit, so a grant given
   later is honoured at the next submit.

### 3.3 Who queries it

Code generation (narrowing barriers, unroll) asks the budget for the kernel,
and the source memo is keyed on the policy (3.2 rule 5). The matrix decision, the
epilogue decision and the coopmat gates read the budget of their kernel (3.1a).
Explicit calls (`Sarek_gemm.Coopmat`, `Sarek_matrix.run`) check their deviations
through the kernel's budget only, without consent, as #440 specifies. The
per-launch cost of the memo is one atomic read while the policy is unchanged.

### 3.4 Environment (SPOC: restrictions only, no aliases)

A replaced switch is removed in M3, not kept as an alias (Q6; nothing is
published yet). The new variable is one parser (`Sarek_policy.Env.parse`: comma
list, any case, spaces and empty items ignored, unknown item warns once on
stderr and is ignored). It is read once, when `Sarek_policy` is initialised
(program start), and acts through `restrict` and an internal revocation. It can
only take away: no variable grants precision (Q3).

| Replaced variable | Policy equivalent | Fate |
|---|---|---|
| `SAREK_LAUNCH_WINDOW=0` | `SAREK_OPT_DISABLE=window` | removed (M3) |
| `SAREK_LAUNCH_EPILOGUE=0` | `SAREK_OPT_DISABLE=epilogue` | removed (M3) |
| `SAREK_MATRIX_REWRITE=off` | `SAREK_OPT_DISABLE=matrix` | removed (M3) |
| `SAREK_MATRIX_REWRITE=exact` | `SAREK_OPT_DISABLE=automatic-float-rewrite` | removed (M3); revokes the consent, so `Execute.run_vectors` rewrites int32 only. Unlike the old mode it does not stop an explicit `Sarek_matrix.run`, which never needed the consent |
| `SAREK_AUTO_UNROLL=0` | `SAREK_OPT_DISABLE=unroll` | removed (M3) |
| `SAREK_MATRIX_REPORT=1` | `SAREK_EXPLAIN=1` | removed (M3) |
| `Matrix_specialise.set_mode` / `mode`, `Sarek_matrix.set_mode` / `mode` | `Sarek_policy.restrict` (one-way, tests) | removed (M3) |
| `Matrix_specialise.set_reporter` | `Sarek_policy.set_reporter` | removed (M3) |
| `SAREK_HIP_IGPU_UNCACHED=0` | not in the policy (U: unsafe, repro only) | unchanged |
| tuning (rows 4, 17-19) | not in the policy (no semantic effect) | unchanged |

`SAREK_OPT_DISABLE` items: `window,epilogue,matrix,narrowing-proof,unroll`
(`Sarek_policy.optimisation_name`) restrict that optimisation;
`accumulation-order`, `operand-rounding-f16|bf16|bf16x3`, `single-rounding`,
`cross-kernel-contraction`, `automatic-float-rewrite` (`Sarek_policy.grant_name`)
revoke that grant for the whole process, at any scope: the budget never holds
it, whatever the program grants. `grant` of a revoked atom still returns
`Ok ()` (the program's statement is recorded), is never effective, and the
explain report says so. There is no `all`. `Execute.Window.set_enabled` stays
(Q4); there is no public scoped-restriction API.

### 3.5 Explain report (M3, implemented)

`SAREK_EXPLAIN=1` installs a default reporter writing to stderr one line per
distinct (kernel, target, optimisation, decision); a repeated launch adds none.
Any other value is off (an unknown one warns once). A reporter installed with
`Sarek_policy.set_reporter` replaces it and receives `event`s: `Decision record`,
`Gate` (the explicit coopmat gate) and `Ignored_grant` (below). The line format is fixed, space-separated, with
spaces in names written `_` and an epilogue pair `p + c` written `p+c`
(`Sarek_policy.Env.explain_line`, pure, tested by a golden):

```text
sarek-explain <kernel> <target> <optimisation> applied proof=<text>
sarek-explain <kernel> <target> <optimisation> applied granted=<grant>[,<grant>] reason="<reason>" [lowering="<description>"]
sarek-explain <kernel> <target> <optimisation> applied exact
sarek-explain <kernel> <target> <optimisation> refused <reason>
sarek-explain <kernel> <target> coopmat-gate applied|refused ...
sarek-explain <kernel> - policy refused grant-on-generated-name <grant>
sarek-explain <kernel> - policy refused grant-revoked <grant>
```

The **target** column has one scheme: `<framework>:<device id>` for a decision
about a device (`window`, `matrix`, `coopmat-gate`; e.g. `HIP:0`),
`<framework>` for one about a backend (`epilogue`), `gen:<generator>` for code
generation (`narrowing-proof`, `unroll`; `gen:cuda-c`, `gen:opencl`,
`gen:metal`, `gen:glsl`). No device names. The report keeps one line per
**distinct** (kernel, target, optimisation, decision text): a set of 4096
entries, emptied when full, so a decision that flips back and forth (the window
per prompt chunk) prints once per distinct outcome. The sink is called outside
the lock.

Where each optimisation reports:

| Optimisation | Reported by | Target |
|---|---|---|
| `window` | `Launch_policy.deferred`, per launch of a JIT kernel: applied exact, or refused (the `Window_gate.refusal` variant: switched off, disabled, CPU device, not on the backend, stream launch, structure-of-arrays arguments). `Window_gate.refusal` is the one function behind the launch path, `Execute.Window.defers_device` and this report; no text is built when nobody listens | `<framework>:<id>` |
| `epilogue` | `Launch_epilogue` per producer and consumer, kernel `p + c` | `<framework>` |
| `matrix` | `Matrix_specialise.choose` for every matrix product (int32: `proof`, the lowering and bit-identical modulo 2^32; float: `granted` with the consent, accumulation order and operand rounding it rests on, `reason` = the program's reasons only, `lowering="..."` the description; refused with the reason) | `<framework>:<id>` |
| `coopmat-gate` (an event kind, not an optimisation) | `Launch_policy.coopmat_gate` from `Execute.check_device_capabilities`, for a kernel the program wrote; the interpreter gate is not reported | `<framework>:<id>` |
| `narrowing-proof` | the CUDA-C generator, per kernel that has an f32 to f16 narrowing: `proof=n/m narrowings proven exact`, refused when none is, `granted=single-rounding`, or disabled | `gen:cuda-c` |
| `unroll` | `Sarek_ir_unroll.annotate`, per kernel and generator, when a loop would be hinted | `gen:<generator>` |

The last two forms of the format come from `Ignored_grant`: a program `Kernel`
grant whose name is generated (a fused, derived or static-fusion kernel, which
never counts, rule 3 of 3.2) and a grant revoked by the environment
(`(process)` for a process-scope one). Each is reported once, right after the
next report for that kernel (also if its decisions settled before), so a user learns that the grant has no
effect. The codegen device column is the generator, not a device: code
generation does not know it.

### 3.6 Downstream engine

- **A precision setting `exact|fast`** (default `exact`). `fast` grants, at
  `Kernel` scope for the engine's own kernels only: `Accumulation_order` (WMMA
  prompt matmuls, today's row 26) and `Single_rounding` for the named dequantise
  kernels (#480: Q4_K keeps its barriers by default; the grant is the lever,
  ~870 to ~170 µs microbenchmark). A profile may also grant
  `Automatic_float_rewrite` together with accuracy atoms (Q1). `fast` also
  selects `Prefill_mma`, whose own f16 rounding of weights and activations
  the engine reports, since SPOC cannot see it.
- **The prefill-MMA switch** is removed when the precision setting lands, with no
  deprecation period (Q6).
- **Memory:** only the RAM-register switch (and the planned CPU executor) are
  resource contracts; the rest is tier strategy with measured trade-offs
  (downstream measurements). A memory setting comes with the CPU expert executor, the second
  resource contract (Q5). The always-pinned staging buffer (row 28) is the engine's
  explicit choice and should appear in the report.
- **The engine's bench report** gains a `policy` section: profile, grants with
  scope and reason, applied/refused counts per optimisation, restrictions in
  force, resource contracts in force (pinned and registered bytes), and every
  `SAREK_*`/`SPOC_*` variable set, not only its own.

## 4. Migration

1. **SPOC M1, no behaviour change (this change):** `Sarek_policy` over the
   existing grants; `accept_relaxed_accumulation`, `accept_operand_rounding`,
   `accept_automatic_float_rewrite` and `Sarek_f16_policy.accept_single_rounding`
   become wrappers of `grant` with the same refusals and messages. Gate: a
   golden of every pure generator's source, of the coopmat verdicts and of the
   grant readers, in five process states, captured on `main` before the change
   and unchanged by it; plus the laws of `Budget`.
2. **SPOC M2 (implemented):** window, epilogue, matrix, narrowing, unroll read
   the effective policy; policy in the source memo; fused-kernel meet; epilogue
   fusion on CUDA, OpenCL and Vulkan under `Cross_kernel_contraction`, measured
   per backend (one contraction per element, `specs/epilogue-fusion.md` FR-107;
   HIP and Metal are exact); CHANGES entry.
3. **SPOC M3 (implemented):** `SAREK_OPT_DISABLE`, `SAREK_EXPLAIN`; the
   replaced switches (`SAREK_LAUNCH_WINDOW`, `SAREK_LAUNCH_EPILOGUE`,
   `SAREK_MATRIX_REWRITE`, `SAREK_AUTO_UNROLL`, `SAREK_MATRIX_REPORT`) and the
   matrix mode and reporter APIs are removed outright;
   `gh-pages/docs/environment.md` completed.
4. **Downstream adoption** (once M2 is the pinned SPOC revision): the precision
   setting, the prefill-MMA switch removed, report section.
5. The `accept_*` wrappers stay.

## 5. Test strategy

- **Behaviour preservation (M1, no GPU).**
  `sarek/tests/codegen_golden/test_policy_golden.ml` prints, for named kernels
  (a provable f16 narrowing, a non-provable Q4_K-like one, an auto-unrolled
  loop, f16 and int32 cooperative-matrix kernels, a plain f32 kernel), the
  source of the CUDA-C (HIP and CUDA/C), OpenCL, Metal, GLSL, WGSL and PTX
  generators, `Sarek_coopmat.policy_verdict` for eight configurations, every
  reader and every refusal message, in five sequential process states (no
  grant; `Kernel` scope; `All_kernels`; the coopmat grants). The expected file
  was produced on `main` and is unchanged by M1.
- **Laws (M1).** `spoc/ir/test/test_sarek_policy.ml`: over all 2^7 subsets of
  the seven atoms, `exact` is bottom, `leq` is a preorder, `meet` is the
  greatest lower bound, `permits` is monotone, `digest` is equal iff the grant
  set is, whatever the reasons and the insertion order; no grant implies
  another; `budget ~kernels` is the meet; the `sarek_kern` and blank refusals.
- **Policy matrix (M2, no GPU).** Fixtures: f16 GEMM, int32 GEMM, a T2
  narrowing, a non-provable Q4_K narrowing, an exact epilogue pair. Fake
  devices: HIP gfx11, CUDA, interpreter. All 2^7 budgets times the restriction
  states. The `Applied` set must equal an independently written table; a
  mutation making one grant enable another must fail it (as #440's
  `~explicit:true` mutation did).
- **Properties (M2).** `leq b b'` implies applied(b) ⊆ applied(b'); a
  restriction only removes; a fused kernel never applies a deviation a member
  lacks.
- **Cache staleness (M2, implemented).** Grant after generating a kernel:
  regenerated source differs (`sarek/execute/test/test_launch_memo.ml`).
- **Env parser (M3, pure, implemented).** `spoc/ir/test/test_sarek_policy_env.ml`: a
  table of inputs, every name, duplicates, unknown items; over all 2^12 subsets of
  items, `Env.narrow` never adds a grant, removes a restriction or turns a refused
  decision into an applied one. `sarek/tests/policy_env`: one process per
  scenario with the variable set by the dune rule (window, epilogue, matrix,
  narrowing-proof, unroll; the grants revoked; an unknown item; the five removed
  variables set and changing nothing).
- **Explain (M3, implemented):** `sarek/tests/policy_env/explain.expected` is the
  golden of the lines for a fixed CPU scenario (codegen, window, matrix, coopmat
  gate, a grant on a generated name); `sarek/execute/test/test_launch_window.ml`
  checks the epilogue and window records; the e2e matrix tests read the matrix
  records.
- **Differential (existing, GPU):** `exact` stays bit-identical to the
  interpreter; each grant stays within its documented bound. Numeric checks stay
  differential on the application side (Q2).
- **Downstream engine:** profile parsing; `fast` yields exactly the documented grants
  (reporter capture); bench report has the policy section.

## 6. Open questions

All eight questions of the first version were answered by the owner on
2026-10-12; the answers are the next section and override the text above where
they differ.

**Limits.** Read from source at the cited revisions. Signatures of
`Sarek_policy` are compiled (M1); `OPTIMISATION`, the explain report and the
downstream settings are not. Figures are quoted from downstream measurements and #480,
not measured here.

## Owner decisions (2026-10-12)

- **Q1:** a profile (the downstream `fast`, or a named SPOC profile) may grant `Automatic_float_rewrite` together with accuracy atoms; `SAREK_EXPLAIN` reports every rewrite.
- **Q2:** no numeric bound per grant. Grants are named deviations, each with its documented bound where one exists; numeric checks stay differential on the application side (logits vs llama.cpp).
- **Q3:** SPOC/SAREK environment variables only restrict, never grant precision.
- **Q4:** no public scoped-restriction API. SPOC decides (fused-kernel disk cache, cost model) and keeps a diagnostic switch; `Window.set_enabled` stays until SPOC handles the prompt-phase case itself.
- **Q5:** a downstream memory setting comes with the CPU expert executor (second resource contract), not before.
- **Q6:** no deprecation period: replaced switches (the downstream prefill-MMA switch, `SAREK_MATRIX_REPORT`, …) are removed outright; nothing is published yet.
- **Q7:** epilogue fusion on CUDA/OpenCL/Vulkan runs under the `Cross_kernel_contraction` grant, measured per backend before it becomes a default; HIP and Metal (#505 pragma) are exact.
- **Q8:** moot — #480 is merged; M1 wraps the final `Sarek_f16_policy`.
