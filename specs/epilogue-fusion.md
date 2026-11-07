---
name: roster-spec
type: spec
status: live
feature: Epilogue fusion of element-wise consumers in the launch window
brief: /mnt/ssd-external-2to/spoc-pipeline-logs/dynamic-fusion/intake.md
date: 2026-10-11
version: 1.0.0
---

# Spec — Epilogue fusion

This is the second part of automatic dynamic fusion; the first is
`specs/launch-window.md`. A launch-window producer P is often followed by an
element-wise consumer C that reads P's output at its own index. Examples:

- a matvec followed by the residual add `x[i] <- x[i] + h[i]`;
- the attention output followed by its sigmoid gate;
- the `moe_combine` output followed by the residual add.

C is merged into P: C's body runs right after each of P's stores, in the
thread that made the store. This saves one launch and one pass over the
intermediate. In a downstream LLM inference engine's decode it removes 54 launches per token on the 0.8B
model and 84 on the 35B.

## Clarifications

| Q | A |
|---|---|
| What exactly is fused? | P's store `I[e] <- v` becomes `let t = e in I[t] <- v; if t < tc then (let i = t in C_body)`, where `tc` is the number of threads C was launched with. C then reads `I[t]` from memory, written just before by the same thread. |
| Why is it equal to running P then C? | Three conditions make it equal, and each is checked before merging. (1) For these dimensions and scalar values, P stores each element of I at most once, and every element C writes is one that P stores. This is proven on the host: by `Coverage_affine` when every store index and guard is affine in the thread and work-group indices, also through division, modulo, masks and shifts by powers of two of values non-negative over the whole launch (guards not counted), which split an index into digits (as in a warp-per-row matrix-vector product) (no enumeration: the store set of P is shown to be one contiguous interval, each element once), and otherwise by `Index_slice`, which runs only the integer computations the stores depend on, for every thread of both launches. A store index below zero is never accepted. (2) C reads nothing else that P writes, and reads I only at its index. (3) C writes nothing P accesses except I itself, and reads its own outputs only at its index. Under these conditions every value C reads is the value it read when run after P. The same operations run in the same order. |
| Is it bit-identical by construction? | Only where the backend compiles every kernel without contraction. P's store and C's load are now in one function, so a device compiler that contracts `a*b+c` could (1) forward P's stored value into C's add, fusing P's last multiply with it, or (2) share a subexpression that P and C both compute (C recomputes P's `a*b`): a product with two uses is not contracted, while the unfused C contracts it. NVRTC 12.6 PTX shows forwarding (1) (SASS not inspected); (2) reproduced on RADV NAVI31 (4099 of 4099 elements differ) and seen in NVRTC PTX by the reviewer. HIP compiles with `-ffp-contract=off` forced last (`Hip_rtc.base_options`), so fusion is on by default on HIP and Metal only. Metal compiles every kernel, fused ones included, with `#pragma METAL fp contract(off)` (`Sarek_ir_metal.metal_fp_contract_pragma`; on an Apple M4 it turns 8773 of 8773 contracted `a*b+c` into 0, docs/fp-contraction-policy.md section 10), so it is on by default as well; `test_epilogue_metal_pragma` checks the pragma in the epilogue-fused and the epilogue-then-horizontal MSL on any host, and the fused == unfused == Interpreter comparison was executed on an Apple M4 (macOS 15): 0 of 4099 elements differ on the forwarding, shared and chain pairs. CUDA (nvrtc `--fmad=true`), OpenCL (contraction on by default; rusticl contracts, executed) and Vulkan fuse only kernels that hold the `Cross_kernel_contraction` grant (FR-107), a precision grant; `Execute.Window.set_epilogue_backend` does not give it. A volatile read of the intermediate was tried: it removes (1) only, so it was dropped (`docs/fp-contraction-policy.md`, section 14). That is the owner rule: bit-identical by construction may be on by default, anything else needs a grant. |
| What is a consumer? | A kernel of the form `let i = global_thread_id in body`. `body` has no barrier, fence, return, `match`, atomic, shared or local array, other position intrinsic or inline native code. It stores only at index `i`, and does not use a vector whole. It must also pass the horizontal-fusion verdict: stdlib intrinsics only, no free variables, no shared memory. C must be 1-D with no dynamic shared memory. |
| Which producers? | Any pending launch, already fused or not, that binds I to exactly one parameter and never reads I. Its stores to I must be plain element stores (`LArrayElem`). |
| When is the coverage evaluated, and what does it cost? | When C arrives, once per (P IR, C IR, both launch shapes, values of the scalars the evaluation read). Scalars the stores do not depend on, such as the attention position, are not part of the key. The affine proof costs microseconds whatever the launch size. Enumeration is bounded: if the two launches have more than 2^20 threads together, an evaluation runs more than 2^21 threads plus loop iterations, or an index or guard depends on data, there is no merge. |
| Interaction with horizontal fusion | The merged launch replaces P in its group. It can host a further epilogue, which gives chains, and it can join or stay in a horizontal group. C must not conflict with P's group mates, nor with any launch of a later group. |
| Failure | If the merged kernel fails to build or launch, P and C are issued as submitted and the pair is never merged again. |

## Functional Requirements

- **FR-101**: A launch C MAY be merged into pending launch P only if all the conditions in the Clarifications hold, on a backend where epilogue fusion is enabled.
- **FR-102**: The merged kernel MUST compute C's element `t` after P's store of `I[t]`, in the same thread, reading `I[t]` from memory, with C's body otherwise unchanged and P's body unchanged apart from binding the store index to a variable.
- **FR-103**: The coverage verdict MUST be computed from these launches' dimensions and scalar values. An unknown result MUST mean no merge.
- **FR-104**: Epilogue fusion MUST need no grant only on a backend that compiles every kernel without floating-point contraction: today HIP and Metal (every Metal kernel carries `#pragma METAL fp contract(off)`; `Sarek_policy.requires Epilogue_fusion` is `Some []`). Metal qualifies only while its conformant compile options (non-fast math) could be built: if `mtl_compile_options_conformant` returns `None`, the plugin marks Metal inexact (`Fp_exactness.mark_inexact`) at start-up, with a warning, and `Launch_epilogue.enabled "Metal"` is false for the process, with or without the grant (fast math is more than contraction). `SAREK_OPT_DISABLE=epilogue` MUST restrict epilogue fusion for the whole process, for good (`Sarek_policy.restrict Epilogue_fusion`; no later `set_epilogue_backend true` brings it back, checked by `test_epilogue_env_off`), and `Execute.Window.set_epilogue_backend ~backend false` on one; `true` MUST only undo a switch-off and MUST NOT grant anything.
- **FR-107**: On a backend that compiles with contraction (CUDA/PTX, CUDA/C, CUDA, OpenCL, Vulkan), a pair is merged only if `Sarek_policy.decide` allows `Epilogue_fusion` for the meet of the budgets of P and C (`Cross_kernel_contraction` held by both, at kernel or process scope; a fused P stands for its members, `Sarek_policy.register_fused`). The decision is taken at every submit, so a grant given after a refusal is honoured by the next submit; the fused-kernel caches hold built kernels only. The merged result differs from the unfused one by at most one contraction per element (for `o = a*b + c`: |f - u| <= ulp(a*b)/2 + ulp(r), where u is the unfused result round32(round32(a*b) + c), f the fused one round32(a*b + c) and r the larger of |f| and |u|; this is a rounding error of the product, so it can be many ulps of a small result when `a*b` and `c` nearly cancel, and 1 ulp of the result only without cancellation; for the residual `a*b - t` it is at most half an ulp of the product); it is not bit-identical in general (`sarek/tests/unit/test_contraction_bound.ml` checks the bound on the host, cancellation included). Native, Interpreter and WebGPU never merge.
- **FR-105**: If the merged kernel fails, P and C MUST be issued as submitted, and the pair MUST NOT be merged again.

## Acceptance Criteria and Checks

- AC-1 [FR-101, FR-102]: A covered pair is merged into one launch. On HIP and Metal the results are bit-identical to the unfused run and to the Interpreter (`test_launch_window`, "dependent"; `test_epilogue_exact`: the forwarding pair `t=a*b; o=t+c`, the shared pair `t=a*b; o=a*b-t`, a chain and a float16-intermediate pair (asserted on the default backends that lower float16, HIP and Metal; OpenCL and Vulkan refuse it), on inputs where an fma differs from multiply-then-add in 1109 of 4099 elements, asserted on the default backends and only recorded on opt-in ones; it cannot detect deviations on other shapes or on hardware absent from the host). On the Interpreter, the merged IR gives bit-identical results to the pair (`test_launch_window`, "epilogue IR = pair").
- AC-2 [FR-103]: An uncovered consumer (domain larger than the producer's stores) is not merged, and its results stay correct ("uncovered consumer" on every device; the probe test "not merged when uncovered"). Mutation check: making the coverage check always pass turns both red.
- AC-3 [FR-104, FR-107]: The decision is checked by a probe test (`test_epilogue_defaults`, through `Launch_epilogue.decide_pair`): HIP and Metal fuse without a grant; CUDA/PTX, CUDA/C, OpenCL and Vulkan do not, even switched on, until both kernels hold the grant; Native, Interpreter and WebGPU never fuse; a backend marked inexact never fuses. `test_launch_window` ("needs the grant") checks it end to end on a probe backend, and that a grant given after a first refused submit is honoured by the next one.
- AC-5 [FR-107]: On OpenCL, Vulkan and CUDA/PTX (ZLUDA) with the grant, `test_epilogue_exact` asserts the fused path was taken (one launch instead of two), prints the number of differing elements and the largest difference, and asserts it within that bound (forwarding, residual, chain); on HIP and Metal it asserts bit equality.
- AC-4: downstream engine decode on the RX 7900 XTX: dispatch counts, and logits bit-identical with and without epilogue fusion.

Checks:
- CHECK-1: `dune build @sarek/execute/test/runtest`.
- CHECK-2: `dune exec sarek/tests/e2e/test_launch_window.exe`, which runs under `dune runtest` on every enumerated device.
- CHECK-5: `dune build @sarek/tests/unit/runtest` (`test_epilogue_defaults`).
- CHECK-4: `dune exec sarek/tests/e2e/test_epilogue_exact.exe` (also with `LD_LIBRARY_PATH=$HOME/opt/zluda`: CUDA then runs through ZLUDA on an AMD GPU, not on NVIDIA hardware); `dune exec sarek/tests/e2e/bench_epilogue.exe` is a hand-run benchmark.
- CHECK-3: `@sarek-hip/test/e2e-hip`, `dune build`, `dune build @fmt`.

## Claims Metadata

```claims
{"record":"claims-header","schema_version":1,"namespace":"epilogue-fusion","spec_lifecycle":"draft"}
{"record":"requirement","id":"FR-101","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-102","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-103","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-104","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-105","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"requirement","id":"FR-107","lifecycle":"draft","external_sources":[],"depends_on":[]}
{"record":"acceptance-criterion","id":"AC-1","for":["FR-101","FR-102"]}
{"record":"acceptance-criterion","id":"AC-2","for":["FR-103"]}
{"record":"acceptance-criterion","id":"AC-3","for":["FR-104","FR-107"]}
{"record":"acceptance-criterion","id":"AC-4","for":["FR-102"]}
{"record":"acceptance-criterion","id":"AC-5","for":["FR-107"]}
{"record":"check","id":"CHECK-1","for":["AC-2","AC-3"]}
{"record":"check","id":"CHECK-2","for":["AC-1","AC-2"]}
{"record":"check","id":"CHECK-3","for":["AC-1"]}
{"record":"check","id":"CHECK-4","for":["AC-1"]}
{"record":"check","id":"CHECK-5","for":["AC-3"]}
```

## Review outcomes (2026-10-11)

The first review returned NO-GO. Two findings were blocking:

- The coverage cache key did not include the producer array. A producer
  that stores two outputs under different guards could have an uncovered
  consumer merged.
- A consumer's helper functions were not scanned.

The review also raised serious issues:

- **Id collisions.** Chained epilogues and horizontal members could collide
  on variable ids, which the interpreter and `Index_slice` both key on.
- **Duplicate declarations.** Two store sites redeclared `ep_t`.
- **Untracked stores.** Stores to the target through a record field or a
  cooperative-matrix tile were not seen.
- **No work bound** on the host evaluation.

All are fixed, and each has a probe test. These include:
- a P→C1→C2 chain on the interpreter, which goes red with the old fixed ids;
- coverage per array;
- duplicate stores;
- scoped store sites;
- disjoint horizontal id ranges;
- the slice limits.

A producer that passes the intermediate to a helper function uses the vector
whole. `Kernel_access` counts that as a read, so the merge is refused. Stores
made inside producer helpers therefore never go unseen.

The re-review returned GO.
